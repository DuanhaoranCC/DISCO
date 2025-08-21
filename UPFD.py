# -*- coding: utf-8 -*-
# @Author  : Alisa
# @File    : main(pretrain).py
# @Software: PyCharm
import warnings
from evaluate import train_test_split
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pargs import pargs
from load_data import TreeDataset, HugeDataset, TreeDataset_PHEME, TreeDataset_UPFD, CovidDataset
from model import UPFD_Net
from torch_geometric import seed_everything

warnings.filterwarnings("ignore")


def evaluate_accuracy(model, eval_loader, device):
    model.eval()
    correct = 0

    for data in eval_loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)  # Predicted class
        correct += pred.eq(data.y).sum().item()  # Only consider validation nodes

    return correct / len(eval_loader.dataset)


def train_with_early_stopping(model, loader, eval_loader, optimizer, device, epochs, patience=10):
    best_val_accuracy = -1
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for data in loader:
            optimizer.zero_grad()
            data = data.to(device)

            out = model(data)
            loss = F.nll_loss(out, data.y)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(loader)

        val_accuracy = evaluate_accuracy(model, eval_loader, device)
        # print(f"Epoch: {epoch}, Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Early stopping triggered. No improvement in validation accuracy for {patience} epochs.")
                break


def evaluate_metrics(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1).cpu().numpy()
            scores = torch.exp(out).cpu().numpy()  # Convert log_softmax to probabilities
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(preds)
            y_scores.extend(scores)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # y_true = np.expand_dims(y_true, axis=1)
    # y_pred = np.expand_dims(y_pred, axis=1)
    y_scores = np.array(y_scores)

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    auc = roc_auc_score(y_true=F.one_hot(torch.from_numpy(y_true)).detach().cpu().numpy(),
                        y_score=y_scores, multi_class='ovr')

    return macro_f1, micro_f1, auc


if __name__ == '__main__':
    args = pargs()
    seed_everything(0)
    dataset = args.dataset
    device = torch.device(f'cuda:{3}' if torch.cuda.is_available() else 'cpu')

    batch_size = 32
    weight_decay = args.weight_decay
    epochs = args.epochs

    # data = TreeDataset("./Data/DRWeiboV3/")
    # data = TreeDataset("./Data/Weibo/")
    data = CovidDataset("./Data/Twitter-COVID19/Twittergraph")
    # data = TreeDataset_PHEME("./Data/pheme/")
    # data = TreeDataset_UPFD("./Data/gossipcop/")
    # data = TreeDataset_UPFD("./Data/politifact/")
    # data = CovidDataset("./Data/Weibo-COVID19/Weibograph")

    # for r in [1, 5]:
    #     mask = train_test_split(data.y.cpu().numpy(), seed=0,
    #                             train_examples_per_class=r,
    #                             val_size=500, test_size=None)
    #     train_mask = mask['train'].astype(bool)
    #     val_mask = mask['val'].astype(bool)
    #     test_mask = mask['test'].astype(bool)
    #
    #     # Create DataLoaders using only the respective masks for each set
    #     train_loader = DataLoader(data[train_mask], batch_size=batch_size, shuffle=True)
    #     val_loader = DataLoader(data[val_mask], batch_size=batch_size, shuffle=False)
    #     test_loader = DataLoader(data[test_mask], batch_size=batch_size, shuffle=False)
    #
    #     # Run the experiment 10 times and collect metrics
    #     micro_f1_scores = []
    #     macro_f1_scores = []
    #     auc_scores = []
    #
    #     for _ in range(10):  # Repeat 10 times
    #         model = UPFD_Net(data.num_features, args.out_feat, data.num_classes).to(device)
    #         optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    #
    #         train_with_early_stopping(model, train_loader, val_loader, optimizer, device, epochs, patience=10)
    #
    #         macro_f1, micro_f1, auc = evaluate_metrics(model, test_loader, device)
    #         macro_f1_scores.append(macro_f1)
    #         micro_f1_scores.append(micro_f1)
    #         auc_scores.append(auc)
    #
    #     # print(f"Run {run + 1}: Macro-F1 = {macro_f1:.4f}, Micro-F1 = {micro_f1:.4f}, AUC = {auc:.4f}")
    #
    #     print("\nFinal Results:")
    #     print(f"Macro-F1: Mean = {np.mean(macro_f1_scores):.4f}, Variance = {np.std(macro_f1_scores):.4f}")
    #     print(f"Micro-F1: Mean = {np.mean(micro_f1_scores):.4f}, Variance = {np.std(micro_f1_scores):.4f}")
    #     print(f"AUC: Mean = {np.mean(auc_scores):.4f}, Variance = {np.var(auc_scores):.4f}")


    # Step 1: Collect all labels (data.y) from the dataset
    all_labels = []
    for idx in range(len(data)):
        sample = data.get(idx)  # Access each sample
        all_labels.append(sample.y.item())  # Extract the label and convert to scalar

    all_labels = np.array(all_labels)  # Convert to a numpy array for train_test_split

    # Step 2: Split data based on labels
    for r in [1, 5]:
        mask = train_test_split(all_labels, seed=0,
                                train_examples_per_class=r,
                                val_size=500, test_size=None)
        train_mask = mask['train'].astype(bool)
        val_mask = mask['val'].astype(bool)
        test_mask = mask['test'].astype(bool)

        # Step 3: Create DataLoaders for each split
        train_data = [data.get(i) for i in range(len(data)) if train_mask[i]]
        val_data = [data.get(i) for i in range(len(data)) if val_mask[i]]
        test_data = [data.get(i) for i in range(len(data)) if test_mask[i]]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Step 4: Continue with training and evaluation
        micro_f1_scores = []
        macro_f1_scores = []
        auc_scores = []

        for _ in range(10):  # Repeat 10 times
            model = UPFD_Net(data.num_features, args.out_feat, data.num_classes).to(device)
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

            train_with_early_stopping(model, train_loader, val_loader, optimizer, device, epochs, patience=10)

            macro_f1, micro_f1, auc = evaluate_metrics(model, test_loader, device)
            macro_f1_scores.append(macro_f1)
            micro_f1_scores.append(micro_f1)
            auc_scores.append(auc)

        # Step 5: Calculate metrics
        print(f"Results for r={r}:")
        print(f"Micro F1 Standard Deviation: {np.mean(micro_f1_scores):.4f}, {np.std(micro_f1_scores):.4f}")
        print(f"Macro F1 Standard Deviation: {np.mean(macro_f1_scores):.4f}, {np.std(macro_f1_scores):.4f}")
        print(f"AUC Standard Deviation: {np.mean(auc_scores):.4f}, {np.std(auc_scores):.4f}")

# Final Results:
# Macro-F1: Mean = 0.5517, Variance = 0.0115
# Micro-F1: Mean = 0.5709, Variance = 0.0047
# AUC: Mean = 0.5876, Variance = 0.0000
#
# Final Results:
# Macro-F1: Mean = 0.3727, Variance = 0.0374
# Micro-F1: Mean = 0.5202, Variance = 0.0056
# AUC: Mean = 0.5157, Variance = 0.0004

# add un direction
# Final Results:
# Macro-F1: Mean = 0.4123, Variance = 0.0687
# Micro-F1: Mean = 0.4793, Variance = 0.0175
# AUC: Mean = 0.4765, Variance = 0.0002
#
# Final Results:
# Macro-F1: Mean = 0.4186, Variance = 0.0806
# Micro-F1: Mean = 0.4984, Variance = 0.0258
# AUC: Mean = 0.5378, Variance = 0.0007

# max un direction
# Final Results:
# Macro-F1: Mean = 0.4416, Variance = 0.0116
# Micro-F1: Mean = 0.4750, Variance = 0.0020
# AUC: Mean = 0.4754, Variance = 0.0000
#
# Final Results:
# Macro-F1: Mean = 0.4873, Variance = 0.0472
# Micro-F1: Mean = 0.5276, Variance = 0.0062
# AUC: Mean = 0.5342, Variance = 0.0001
