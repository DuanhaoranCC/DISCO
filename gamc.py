# -*- coding: utf-8 -*-
# @Author  : Alisa
# @File    : main(pretrain).py
# @Software: PyCharm
import warnings
from evaluate import evaluate, train_test_split
import numpy as np
import itertools
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pargs import pargs
from load_data import TreeDataset, HugeDataset, TreeDataset_PHEME, TreeDataset_UPFD, CovidDataset, load_datasets_with_prompts
from model import GAMC
from augmentation import augment
from torch_geometric import seed_everything
from torch_geometric.nn import global_add_pool
warnings.filterwarnings("ignore")


def pre_train_individual(loader, model, optimizer, device):
    """
    Pre-train the model with a single DataLoader.
    - Apply augmentations and compute loss for a single dataset
    """
    model.train()
    total_loss = 0

    # Loop through each data batch from the loader
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)

        loss = model(data, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (data.num_graphs)

    return total_loss / len(loader.dataset)


def pre_train(loaders, model, optimizer, device):
    """
    Pre-train the model with multiple DataLoaders.

    :param loaders: List of DataLoaders for the datasets.
    :param model: The model to train.
    :param optimizer: Optimizer for the training process.
    :param device: Device to perform computations on (e.g., 'cuda' or 'cpu').
    :return: Average loss over all datasets.
    """
    model.train()
    total_loss = 0

    # Iterate through batches from each DataLoader, using itertools.zip_longest to handle different lengths
    for i, batches in enumerate(itertools.zip_longest(*loaders, fillvalue=None)):
        optimizer.zero_grad()

        augmented_data1 = []

        # Process each batch from the different loaders
        for idx, batch in enumerate(batches):
            if batch is not None:  # Ensure the batch is not None (handle shorter datasets)
                batch = batch.to(device)
                augmented_data1.append(batch)

        loss = model(*augmented_data1)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    total_loss /= len(loaders)
    return total_loss


if __name__ == '__main__':
    args = pargs()
    seed_everything(0)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    batch_size = 32
    lr = 0.00015
    weight_decay = 0.0
    epochs = 80

    # Initialize datasets
    # data = TreeDataset("./Data/DRWeiboV3/")
    # data = TreeDataset("./Data/Weibo/")
    # data = TreeDataset("./Data/Twitter15-tfidf/")
    # data = TreeDataset("./Data/Twitter16-tfidf/")
    # data = TreeDataset_PHEME("./Data/pheme/")
    # data = TreeDataset_UPFD("./Data/politifact/")
    # data = TreeDataset_UPFD("./Data/gossipcop/")
    # data = CovidDataset("./Data/Twitter-COVID19/Twittergraph")
    # data = CovidDataset("./Data/Weibo-COVID19/Weibograph")


    # Create DataLoaders
    # loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    #
    # eval_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    loader, eval_loader = load_datasets_with_prompts(args)

    # Model and optimizer initialization
    model = GAMC(768, 128).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # for epoch in range(1, epochs + 1):
    #     # pretrain_loss = pre_train_individual(loader, model, optimizer, device)
    #     pretrain_loss = pre_train(loader, model, optimizer, device)
    #     print(f"Epoch: {epoch}, loss: {pretrain_loss}")
    # torch.save(model.state_dict(), f"./{args.dataset}_gamc.pt")
    model.load_state_dict(torch.load(f"./{args.dataset}_gamc.pt", map_location=device))

    # Evaluation
    model.eval()
    x_list = []
    y_list = []
    for data in eval_loader:
        data = data.to(device)
        embeds = model.get_embeds(data).detach()
        # embeds = global_add_pool(data.x, data.batch).detach()
        ################################
        # root_indices = []
        # batch_size = max(data.batch) + 1
        # for num_batch in range(batch_size):
        #     root_indices.append(torch.nonzero(data.batch == num_batch, as_tuple=False)[0].item())
        # embeds = data.x[root_indices]
        ################################
        y_list.append(data.y)
        x_list.append(embeds)
    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)

    for r in [1, 5]:
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
# Macro-F1_mean: 0.5623 var: 0.0162  Micro-F1_mean: 0.6174 var: 0.0000 auc 0.5050 var: 0.0143
# Macro-F1_mean: 0.5795 var: 0.0155  Micro-F1_mean: 0.6217 var: 0.0112 auc 0.6255 var: 0.0087

# Pheme
# Macro-F1_mean: 0.4838 var: 0.0495  Micro-F1_mean: 0.6238 var: 0.0323 auc 0.5657 var: 0.0370
# Macro-F1_mean: 0.5908 var: 0.0081  Micro-F1_mean: 0.6312 var: 0.0006 auc 0.5631 var: 0.0195

# Weibo
# 0.001
# Macro-F1_mean: 0.4300 var: 0.0379  Micro-F1_mean: 0.5093 var: 0.0047 auc 0.5001 var: 0.0163
# Macro-F1_mean: 0.5233 var: 0.0492  Micro-F1_mean: 0.5302 var: 0.0410 auc 0.5320 var: 0.0799
# 0.00015
# Macro-F1_mean: 0.4235 var: 0.0299  Micro-F1_mean: 0.5180 var: 0.0090 auc 0.5192 var: 0.0367
# Macro-F1_mean: 0.5614 var: 0.0335  Micro-F1_mean: 0.5673 var: 0.0330 auc 0.5810 var: 0.0482

# DRWeibo
# 0.001
# Macro-F1_mean: 0.5881 var: 0.0151  Micro-F1_mean: 0.5872 var: 0.0166 auc 0.6138 var: 0.0212
# Macro-F1_mean: 0.5804 var: 0.0308  Micro-F1_mean: 0.5754 var: 0.0217 auc 0.5943 var: 0.0294

# 0.00015
# Macro-F1_mean: 0.5461 var: 0.0262  Micro-F1_mean: 0.5580 var: 0.0130 auc 0.6069 var: 0.0268
# Macro-F1_mean: 0.5423 var: 0.0182  Micro-F1_mean: 0.5472 var: 0.0137 auc 0.5394 var: 0.0254


# politifact
# Macro-F1_mean: 0.5315 var: 0.0492  Micro-F1_mean: 0.5604 var: 0.0377 auc 0.6517 var: 0.0429
# Macro-F1_mean: 0.6885 var: 0.0426  Micro-F1_mean: 0.6809 var: 0.0542 auc 0.7765 var: 0.0368

# gossipcop
# Macro-F1_mean: 0.6555 var: 0.0108  Micro-F1_mean: 0.6705 var: 0.0092 auc 0.5924 var: 0.0154
# Macro-F1_mean: 0.7966 var: 0.0133  Micro-F1_mean: 0.7938 var: 0.0142 auc 0.8279 var: 0.0227

# Weibo-COVID19
# Macro-F1_mean: 0.5874 var: 0.0219  Micro-F1_mean: 0.6135 var: 0.0081 auc 0.4411 var: 0.0514
# Macro-F1_mean: 0.5514 var: 0.0106  Micro-F1_mean: 0.6505 var: 0.0000 auc 0.4348 var: 0.0213
