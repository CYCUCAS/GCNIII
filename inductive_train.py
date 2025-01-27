import time
import random
import argparse
import numpy as np

import torch
import torch.optim as optim

from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from model import GCNII_inductive, GCNIII_inductive

from sklearn.metrics import f1_score
import uuid

## Settings
def setup_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='GCNIII', help='Name of model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--layer', type=int, default=9, help='Number of layers.')
    parser.add_argument('--hidden', type=int, default=2048, help='Hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dropedge', type=float, default=0.0, help='DropEdge rate (1 - keep probability).')
    parser.add_argument('--intersect_memory', action='store_true', default=False, help='Intersect memory.')
    parser.add_argument('--initial_residual', action='store_true', default=False, help='Initial residual.')
    parser.add_argument('--identity_mapping', action='store_true', default=False, help='Identity mapping.')
    parser.add_argument('--batchnorm_wide', type=bool, default=True, help='BatchNorm in wide model.')
    parser.add_argument('--patience', type=int, default=2000, help='Patience')
    parser.add_argument('--dataset', default='ppi', help='Dateset')
    parser.add_argument('--dev', type=int, default=0, help='device id')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=1, help='lamda.')
    parser.add_argument('--gamma', type=float, default=0.05, help='gamma.')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    return args

args = setup_params()


## Load Data
device = torch.device("cuda:" + str(args.dev) if torch.cuda.is_available() else "cpu")
checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'

train_dataset = PPIDataset(mode="train")
val_dataset = PPIDataset(mode="valid")
test_dataset = PPIDataset(mode="test")
features = train_dataset[0].ndata["feat"]

if args.model == 'GCNII':
    model = GCNII_inductive(
            in_size=features.shape[1],
            out_size=train_dataset.num_classes,
            hidden_size=args.hidden,
            n_layers=args.layer,
            alpha=args.alpha,
            lamda=args.lamda,
            dropout=args.dropout,
        ).to(device)
elif args.model == 'GCNIII':
    model = GCNIII_inductive(
            in_size=features.shape[1],
            out_size=train_dataset.num_classes,
            hidden_size=args.hidden,
            n_layers=args.layer,
            alpha=args.alpha,
            lamda=args.lamda,
            gamma=args.gamma,
            dropout=args.dropout,
            dropedge=args.dropedge,
            intersect_memory=args.intersect_memory,
            initial_residual=args.initial_residual,
            identity_mapping=args.identity_mapping,
            batchnorm_wide=args.batchnorm_wide,
        ).to(device)
else:
    raise ValueError("Unknown model: {}".format(args.model))

loss_fcn = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

train_dataloader = GraphDataLoader(train_dataset, batch_size=1)
val_dataloader = GraphDataLoader(val_dataset, batch_size=1)
test_dataloader = GraphDataLoader(test_dataset, batch_size=1)


## Train
def evaluate(g, features, labels, model):
    model.eval()
    with torch.no_grad():
        output = model(g, features)
        loss_data = loss_fcn(output, labels)
        pred = np.where(output.data.cpu().numpy() > 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), pred, average="micro")
        return score, loss_data.item()

def evaluate_in_batches(dataloader, device, model):
    total_loss = 0
    total_score = 0
    for batch_id, batched_graph in enumerate(dataloader):
        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata["feat"]
        labels = batched_graph.ndata["label"]
        score, val_loss = evaluate(batched_graph, features, labels, model)
        total_loss += val_loss
        total_score += score
    return total_loss / (batch_id + 1), total_score / (batch_id + 1)

def train():
    model.train()
    loss_tra = 0
    acc_tra = 0
    for step, batched_graph in enumerate(train_dataloader):
        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata["feat"].float()
        labels = batched_graph.ndata["label"].float()
        logits = model(batched_graph, features)
        loss = loss_fcn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tra += loss.item()
    loss_tra /= 20
    acc_tra /= 20
    return loss_tra, acc_tra


def validation():
    loss_val, acc_val = evaluate_in_batches(val_dataloader, device, model)
    return loss_val, acc_val


def test():
    model.load_state_dict(torch.load(checkpt_file))
    loss_test, acc_test = evaluate_in_batches(test_dataloader, device, model)
    return loss_test, acc_test


t_total = time.time()
bad_counter = 0
acc = 0
best_epoch = 1
for epoch in range(args.epochs):
    loss_tra, acc_tra = train()
    loss_val, acc_val = validation()

    if (epoch + 1) % 1 == 0:
        print('Epoch:{:04d}'.format(epoch + 1),
              'train',
              'loss:{:.3f}'.format(loss_tra),
              '| val',
              'loss:{:.3f}'.format(loss_val),
              'f1:{:.3f}'.format(acc_val * 100))

    if acc_val > acc:
        acc = acc_val
        best_epoch = epoch
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break


acc = test()[1]

print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch'.format(best_epoch))
print("Test acc: {:.2f}".format(acc * 100))





