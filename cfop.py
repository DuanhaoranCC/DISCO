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
from load_data import TreeDataset, HugeDataset, TreeDataset_PHEME, TreeDataset_UPFD, CovidDataset, \
    load_datasets_with_prompts
from model import CFOP, CosineDecayScheduler
from torch_geometric.data import Data
from torch_geometric import seed_everything
from torch_geometric.nn import global_add_pool

warnings.filterwarnings("ignore")

def pre_train_individual(loader, model, optimizer, device, mm_scheduler, lr_scheduler):
    """
    Pre-train the model with a single DataLoader.
    - Apply augmentations and compute loss for a single dataset
    """
    model.train()
    total_loss = 0

    # update momentum
    mm = 1 - mm_scheduler.get(epoch - 1)
    # mm = 0.99
    # update learning rate
    lr = lr_scheduler.get(epoch - 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Loop through each data batch from the loader
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)

        loss = model(data)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        total_loss += loss.item() * (data.num_graphs)

    return total_loss / len(loader.dataset)


def pre_train(loaders, model, optimizer, device, mm_scheduler, lr_scheduler):
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
    # update momentum
    mm = 1 - mm_scheduler.get(epoch - 1)
    # mm = 0.99
    # update learning rate
    lr = lr_scheduler.get(epoch - 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
        model.update_target_network(mm)

        total_loss += loss.item()
    total_loss /= len(loaders)
    return total_loss


if __name__ == '__main__':
    args = pargs()
    seed_everything(0)
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    batch_size = 32
    # 1e-3
    lr = 1e-5
    # 5e-4, 1e-4
    weight_decay = 1e-5
    epochs = 100

    # Initialize datasets
    # data = TreeDataset("./Data/DRWeiboV3/")
    # data = TreeDataset("./Data/Weibo/")
    # data = TreeDataset("./Data/Twitter15-tfidf/")
    # data = TreeDataset("./Data/Twitter16-tfidf/")
    # data = TreeDataset_PHEME("./Data/pheme/")
    # data = TreeDataset_UPFD("./Data/politifact/")
    # data = TreeDataset_UPFD("./Data/gossipcop/")
    data = CovidDataset("./Data/Twitter-COVID19/Twittergraph")
    # data = CovidDataset("./Data/Weibo-COVID19/Weibograph")

    # Create DataLoaders
    # loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    # eval_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    loader, eval_loader = load_datasets_with_prompts(args)

    # Model and optimizer initialization
    model = CFOP(768, 32, rate=0.5, alpha=0.5).to(device)
    lr_scheduler = CosineDecayScheduler(lr, 100, epochs)
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, epochs)
    # optimizer
    optimizer = Adam(model.trainable_parameters(), lr=lr, weight_decay=weight_decay)


    # for epoch in range(1, epochs + 1):
    #     # pretrain_loss = pre_train_individual(loader, model, optimizer, device, mm_scheduler, lr_scheduler)
    #     pretrain_loss = pre_train(loader, model, optimizer, device, mm_scheduler, lr_scheduler)
    #     print(f"Epoch: {epoch}, loss: {pretrain_loss}")
    # torch.save(model.state_dict(), f"./{args.dataset}_cfop.pt")
    model.load_state_dict(torch.load(f"./{args.dataset}_cfop.pt"))

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

# Weibo-Single 1e-5
# Macro-F1_mean: 0.5238 var: 0.0677  Micro-F1_mean: 0.5552 var: 0.0430 auc 0.5933 var: 0.0677
# Macro-F1_mean: 0.5995 var: 0.0237  Micro-F1_mean: 0.5969 var: 0.0273 auc 0.6064 var: 0.0353
