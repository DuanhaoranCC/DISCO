# -*- coding: utf-8 -*-
# @Author  : Alisa
# @File    : main(pretrain).py
# @Software: PyCharm
import warnings
from evaluate import evaluate, train_test_split
import numpy as np
import time
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pargs import pargs
from load_data import TreeDataset, HugeDataset, TreeDataset_PHEME, TreeDataset_UPFD, CovidDataset
from model import BiGCN_individual
from augmentation import augment
from torch_geometric import seed_everything
from torch_geometric.nn import global_add_pool
warnings.filterwarnings("ignore")


def pre_train_individual(loader, aug1, aug2, model, optimizer, device):
    """
    Pre-train the model with a single DataLoader.
    - Apply augmentations and compute loss for a single dataset
    """
    model.train()
    total_loss = 0

    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    # Loop through each data batch from the loader
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)

        aug_data1 = augment(data, augs1)
        aug_data2 = augment(data, augs2)

        out1 = model(aug_data1)
        out2 = model(aug_data2)
        loss = model.loss_graphcl(out1, out2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (data.num_graphs)

    return total_loss / len(loader.dataset)


if __name__ == '__main__':
    args = pargs()
    seed_everything(0)
    dataset = args.dataset
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

    batch_size = 32

    weight_decay = args.weight_decay
    epochs = args.epochs

    # Initialize datasets
    data = TreeDataset("./Data/DRWeiboV3/")
    # data = TreeDataset("./Data/Weibo/")
    # data = TreeDataset("./Data/Twitter15-tfidf/")
    # data = TreeDataset("./Data/Twitter16-tfidf/")
    # data = TreeDataset_PHEME("./Data/pheme/")
    # data = TreeDataset_UPFD("./Data/politifact/")
    # data = TreeDataset_UPFD("./Data/gossipcop/")
    # data = CovidDataset("./Data/Twitter-COVID19/Twittergraph")
    # data = CovidDataset("./Data/Weibo-COVID19/Weibograph")


    # Create DataLoaders
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    eval_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    # Model and optimizer initialization
    model = BiGCN_individual(data.num_features, args.out_feat, data.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        pretrain_loss = pre_train_individual(loader, args.aug1, args.aug2, model, optimizer, device)
        print(f"Epoch: {epoch}, loss: {pretrain_loss}")
    # torch.save(model.state_dict(), f"./{dataset}_Individual.pt")
    # model.load_state_dict(torch.load(f"./{dataset}_all_0.3.pt"))

    # Evaluation
    model.eval()
    x_list = []
    y_list = []
    for data in eval_loader:
        data = data.to(device)
        # embeds = model.get_embeds(data).detach()
        # embeds = global_add_pool(data.x, data.batch).detach()
        ################################
        root_indices = []
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            root_indices.append(torch.nonzero(data.batch == num_batch, as_tuple=False)[0].item())
        embeds = data.x[root_indices]
        ################################
        y_list.append(data.y)
        x_list.append(embeds)
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)

    for r in [1, 5, 10, 20]:
        mask = train_test_split(y.cpu().numpy(), seed=0,
                                train_examples_per_class=r,
                                val_size=500, test_size=None)
        train_mask_l = f"{r}_train_mask"
        train_mask = mask['train'].astype(bool)
        val_mask_l = f"{r}_val_mask"
        val_mask = mask['val'].astype(bool)

        test_mask_l = f"{r}_test_mask"
        test_mask = mask['test'].astype(bool)

        evaluate(x, y, device, 0.01, 0.0, train_mask, val_mask, test_mask)

# Twitter-Covid19
# Macro-F1_mean: 0.5788 var: 0.0663  Micro-F1_mean: 0.5973 var: 0.0702 auc 0.6299 var: 0.0271
# Macro-F1_mean: 0.5787 var: 0.0495  Micro-F1_mean: 0.6066 var: 0.0642 auc 0.5845 var: 0.0570
# Macro-F1_mean: 0.6072 var: 0.0295  Micro-F1_mean: 0.6457 var: 0.0552 auc 0.6111 var: 0.0269
# Macro-F1_mean: 0.6387 var: 0.0369  Micro-F1_mean: 0.7031 var: 0.0478 auc 0.6325 var: 0.0367

# Pheme
# Macro-F1_mean: 0.5331 var: 0.0438  Micro-F1_mean: 0.5361 var: 0.0452 auc 0.5785 var: 0.0567
# Macro-F1_mean: 0.5946 var: 0.0329  Micro-F1_mean: 0.6106 var: 0.0314 auc 0.5858 var: 0.0934
# Macro-F1_mean: 0.6661 var: 0.0233  Micro-F1_mean: 0.6793 var: 0.0273 auc 0.6901 var: 0.0604
# Macro-F1_mean: 0.6675 var: 0.0123  Micro-F1_mean: 0.6925 var: 0.0068 auc 0.7157 var: 0.0115

# Weibo
# Macro-F1_mean: 0.6191 var: 0.0782  Micro-F1_mean: 0.6438 var: 0.0669 auc 0.6724 var: 0.0766
# Macro-F1_mean: 0.7049 var: 0.0433  Micro-F1_mean: 0.7109 var: 0.0396 auc 0.7406 var: 0.0568
# Macro-F1_mean: 0.7830 var: 0.0185  Micro-F1_mean: 0.7840 var: 0.0177 auc 0.8275 var: 0.0171
# Macro-F1_mean: 0.8117 var: 0.0089  Micro-F1_mean: 0.8117 var: 0.0087 auc 0.8587 var: 0.0053

# DRWeibo
# Macro-F1_mean: 0.5676 var: 0.0286  Micro-F1_mean: 0.5966 var: 0.0167 auc 0.6223 var: 0.0239
# Macro-F1_mean: 0.6239 var: 0.0352  Micro-F1_mean: 0.6292 var: 0.0386 auc 0.6552 var: 0.0445
# Macro-F1_mean: 0.7024 var: 0.0083  Micro-F1_mean: 0.7058 var: 0.0081 auc 0.7484 var: 0.0063
# Macro-F1_mean: 0.6987 var: 0.0047  Micro-F1_mean: 0.7015 var: 0.0074 auc 0.7341 var: 0.0099

# politifact
# Macro-F1_mean: 0.5160 var: 0.0590  Micro-F1_mean: 0.5222 var: 0.0323 auc 0.5637 var: 0.0474
# Macro-F1_mean: 0.6258 var: 0.0452  Micro-F1_mean: 0.6319 var: 0.0428 auc 0.6520 var: 0.0492
# Macro-F1_mean: 0.7073 var: 0.0219  Micro-F1_mean: 0.7062 var: 0.0233 auc 0.7224 var: 0.0240
# Macro-F1_mean: 0.7058 var: 0.0188  Micro-F1_mean: 0.7069 var: 0.0178 auc 0.7385 var: 0.0146

# gossipcop
# Macro-F1_mean: 0.6969 var: 0.0442  Micro-F1_mean: 0.7009 var: 0.0410 auc 0.7141 var: 0.0458
# Macro-F1_mean: 0.8323 var: 0.0329  Micro-F1_mean: 0.8328 var: 0.0326 auc 0.8503 var: 0.0339
# Macro-F1_mean: 0.8637 var: 0.0147  Micro-F1_mean: 0.8637 var: 0.0151 auc 0.8795 var: 0.0148
# Macro-F1_mean: 0.8662 var: 0.0272  Micro-F1_mean: 0.8659 var: 0.0257 auc 0.8856 var: 0.0238

# Weibo-COVID19
# Macro-F1_mean: 0.7133 var: 0.0339  Micro-F1_mean: 0.7239 var: 0.0370 auc 0.7332 var: 0.0502
# Macro-F1_mean: 0.7751 var: 0.0176  Micro-F1_mean: 0.7785 var: 0.0174 auc 0.8082 var: 0.0230
# Macro-F1_mean: 0.7494 var: 0.0291  Micro-F1_mean: 0.7563 var: 0.0354 auc 0.7667 var: 0.0363
# Macro-F1_mean: 0.8083 var: 0.0174  Micro-F1_mean: 0.8286 var: 0.0159 auc 0.8370 var: 0.0193
