from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import GCN, GAT, APPNP, GCNII, GCNIII
import uuid


## Settings
def setup_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='GCNIII', help='Name of model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--wd1', type=float, default=0.01, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--wd2', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dropedge', type=float, default=0.0, help='DropEdge rate (1 - keep probability).')
    parser.add_argument('--intersect_memory', action='store_true', default=False, help='Intersect memory.')
    parser.add_argument('--initial_residual', action='store_true', default=False, help='Initial residual.')
    parser.add_argument('--identity_mapping', action='store_true', default=False, help='Identity mapping.')
    parser.add_argument('--batchnorm_wide', type=bool, default=True, help='Batchnorm in wide model.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--dataset', default='cora', help='Dateset')
    parser.add_argument('--dev', type=int, default=0, help='Device Id')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha_l')
    parser.add_argument('--lamda', type=float, default=0.5, help='Lamda.')
    parser.add_argument('--gamma', type=float, default=0.05, help='Gamma.')
    parser.add_argument('--train', action='store_true', default=False, help='Training on train set.')
    parser.add_argument('--test', action='store_true', default=False, help='Evaluation on test set.')
    parser.add_argument('--pretrain', action='store_true', default=False, help='Evaluation pretrained model on test set.')
    parser.add_argument('--name', type=str, default='', help='Name of pretrained model parameter.')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    return args

args = setup_params()


## Load Data
transform = (AddSelfLoop())
if args.dataset == "cora":
    data = CoraGraphDataset(transform=transform)
elif args.dataset == "citeseer":
    data = CiteseerGraphDataset(transform=transform)
elif args.dataset == "pubmed":
    data = PubmedGraphDataset(transform=transform)
else:
    raise ValueError("Unknown dataset: {}".format(args.dataset))

device = torch.device("cuda:" + str(args.dev) if torch.cuda.is_available() else "cpu")
g = data[0].to(device)
features = g.ndata["feat"]
labels = g.ndata["label"]
masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'

if args.model == 'GCN':
    model = GCN(
        in_size=features.shape[1],
        hid_size=args.hidden,
        out_size=int(labels.max()) + 1,
        n_layers=args.layer,
        dropout=args.dropout,
    ).to(device)
elif args.model == 'GAT':
    heads = []
    for i in range(args.layer - 1):
        heads.append(8)
    heads.append(1)
    model = GAT(
        in_size=features.shape[1],
        hid_size=args.hidden,
        out_size=int(labels.max()) + 1,
        heads=heads,
        n_layers=args.layer,
        dropout=args.dropout,
    ).to(device)
elif args.model == 'APPNP':
    model = APPNP(
        in_size=features.shape[1],
        hid_size=args.hidden,
        out_size=int(labels.max()) + 1,
        n_layers=args.layer,
        alpha=args.alpha,
        dropout=args.dropout,
    ).to(device)
elif args.model == 'GCNII':
    model = GCNII(
        in_size=features.shape[1],
        out_size=int(labels.max()) + 1,
        hidden_size=args.hidden,
        n_layers=args.layer,
        alpha=args.alpha,
        lamda=args.lamda,
        dropout=args.dropout,
    ).to(device)
elif args.model == 'GCNIII':
    model = GCNIII(
        in_size=features.shape[1],
        out_size=int(labels.max()) + 1,
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

if args.model in ['GCNII', 'GCNIII']:
    optimizer = optim.Adam([
        {'params': model.params1, 'weight_decay': args.wd1},
        {'params': model.params2, 'weight_decay': args.wd2},
    ], lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


## Train
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)

def train():
    train_mask = masks[0]
    model.train()
    optimizer.zero_grad()
    output = model(g, features)
    acc_train = accuracy(output[train_mask], labels[train_mask].to(device))
    loss_train = F.nll_loss(output[train_mask], labels[train_mask].to(device))
    loss_train.backward()
    optimizer.step()

    return loss_train.item(), acc_train.item()


def validate():
    val_mask = masks[1]
    model.eval()
    with torch.no_grad():
        output = model(g, features)
        loss_val = F.nll_loss(output[val_mask], labels[val_mask].to(device))
        acc_val = accuracy(output[val_mask], labels[val_mask].to(device))

        return loss_val.item(), acc_val.item()


def test():
    test_mask = masks[2]
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(g, features)
        loss_test = F.nll_loss(output[test_mask], labels[test_mask].to(device))
        acc_test = accuracy(output[test_mask], labels[test_mask].to(device))

        return loss_test.item(), acc_test.item()


def test_from_pretrain(file_name):
    checkpt_file = 'pretrained/' + file_name
    test_mask = masks[2]
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(g, features)
        loss_test = F.nll_loss(output[test_mask], labels[test_mask].to(device))
        acc_test = accuracy(output[test_mask], labels[test_mask].to(device))

        return loss_test.item(), acc_test.item()

if args.train:
    print("Training...")
    time_begin = time.time()
    bad_counter = 0
    best = float('inf')
    best_epoch = 1
    acc = 0
    for epoch in range(args.epochs):
        loss_tra, acc_tra = train()
        loss_val, acc_val = validate()
        if (epoch + 1) % 1 == 0:
            print('Epoch:{:04d}'.format(epoch + 1),
                  'train',
                  'loss:{:.3f}'.format(loss_tra),
                  'acc:{:.2f}'.format(acc_tra * 100),
                  '| val',
                  'loss:{:.3f}'.format(loss_val),
                  'acc:{:.2f}'.format(acc_val * 100))
        if loss_val < best:
            best = loss_val
            best_epoch = epoch + 1
            acc = acc_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    print("Train cost: {:.4f}s".format(time.time() - time_begin))
    print('Load {}th epoch'.format(best_epoch))
    print("Val acc.:{:.1f}".format(acc * 100))

if args.test:
    print("Testing...")
    acc = test()[1]
    print("Test acc.:{:.1f}".format(acc * 100))

if args.pretrain:
    print("Test pretrained model...")
    parameter_id = args.name
    acc = test_from_pretrain(parameter_id)[1]
    print("Test acc.:{:.1f}".format(acc * 100))
