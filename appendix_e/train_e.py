import argparse

import torch
import torch.nn as nn

import dgl
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

from model_e import GCN, GCN_featdrop, GCN_learnfeat

## semi-supervised
def evaluate_semi(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits= model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)

        return correct.item() * 1.0 / len(labels)

def train_semi(g, features, labels, masks, model):
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    for epoch in range(200):
        model.train()
        logits= model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = evaluate_semi(g, features, labels, train_mask, model)
        valid_acc = evaluate_semi(g, features, labels, val_mask, model)

        print(
            "Epoch {:05d} | Loss {:.4f} | Train Accuracy {:.4f} | Valid Accuracy {:.4f}".format(
                epoch, loss.item(), train_acc, valid_acc
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="pubmed",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()

    transform = (
        AddSelfLoop()
    )
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]


    ### Feature ###

    # prob_tensor = torch.full((g.num_nodes(), 1000), 0.1)
    # features = torch.bernoulli(prob_tensor)  # Random(0-1)

    # features = torch.rand(g.num_nodes(), 1000)  # Random(dense)

    # features = torch.eye(g.num_nodes()).to(device)  # One-hot

    features = g.ndata["feat"]  # Bag-of-words

    ###############

    in_size = features.shape[1]
    out_size = data.num_classes

    model = GCN(in_size, 64, out_size).to(device)
    # model = GCN_featdrop(in_size, 64, out_size).to(device)
    # model = GCN_learnfeat(g, in_size, 64, out_size, 1000).to(device)

    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    ## semi-supervised
    print("semi-supervised...")
    # model training
    print("Training...")
    train_semi(g, features, labels, masks, model)

    # test the model
    print("Testing...")
    acc = evaluate_semi(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
