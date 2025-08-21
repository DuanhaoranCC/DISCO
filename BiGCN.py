# -*- coding: utf-8 -*-
# @Author  : Alisa
# @File    : main(pretrain).py
# @Software: PyCharm
import warnings
from evaluate import train_test_split
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from pargs import pargs
from load_data import TreeDataset, HugeDataset, TreeDataset_PHEME, TreeDataset_UPFD, CovidDataset
from model import BiGCN_individual
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric import seed_everything
warnings.filterwarnings("ignore")


def train_with_early_stopping(model, loader, eval_loader, optimizer, device, epochs, patience=10):
    best_val_accuracy = -1  # Initialize best validation accuracy to negative (so it can improve)
    patience_counter = 0  # To track the number of epochs with no improvement

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        # Training loop
        for data in loader:
            optimizer.zero_grad()
            data = data.to(device)

            out = model.bigcn(data)
            loss = F.nll_loss(out, data.y)  # Only compute loss on train_mask
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(loader)

        # Evaluate the model on the validation set (using accuracy for early stopping)
        val_accuracy = evaluate_accuracy(model, eval_loader, device)
        # print(f"Epoch: {epoch}, Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Early stopping logic (based on validation accuracy)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0  # Reset counter if validation accuracy improves
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Early stopping triggered. No improvement in validation accuracy for {patience} epochs.")
                break

    # Load the best model (with the highest validation accuracy)
    # model.load_state_dict(torch.load("best_model.pt"))

def evaluate_test_metrics(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for data in test_loader:
        data = data.to(device)
        out = model.bigcn(data)
        pred = out.argmax(dim=1)  # Predicted class
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    # Calculate Micro F1, Macro F1 and AUC
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    auc = roc_auc_score(all_labels, all_preds, average='macro', multi_class='ovr')

    return micro_f1, macro_f1, auc

def evaluate_accuracy(model, eval_loader, device):
    model.eval()
    correct = 0

    for data in eval_loader:
        data = data.to(device)
        out = model.bigcn(data)
        pred = out.argmax(dim=1)  # Predicted class
        correct += pred.eq(data.y).sum().item()  # Only consider validation nodes

    return correct / len(eval_loader.dataset)

if __name__ == '__main__':
    args = pargs()
    seed_everything(0)
    dataset = args.dataset
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

    batch_size = 32
    weight_decay = args.weight_decay
    epochs = args.epochs

    # Initialize datasets
    # data = TreeDataset("./Data/DRWeiboV3/")
    # data = TreeDataset("./Data/Weibo/")
    # data = TreeDataset("./Data/Twitter15-tfidf/")
    # data = TreeDataset("./Data/Twitter16-tfidf/")
    # data = TreeDataset_PHEME("./Data/pheme/")
    # data = TreeDataset_UPFD("./Data/politifact/")
    data = TreeDataset_UPFD("./Data/gossipcop/")
    # data = CovidDataset("./Data/Twitter-COVID19/Twittergraph")
    # data = CovidDataset("./Data/Weibo-COVID19/Weibograph")

    all_micro_f1 = []
    all_macro_f1 = []
    all_auc = []


    for r in [1, 5]:
        mask = train_test_split(data.y.cpu().numpy(), seed=0,
                                train_examples_per_class=r,
                                val_size=500, test_size=None)
        train_mask = mask['train'].astype(bool)
        val_mask = mask['val'].astype(bool)
        test_mask = mask['test'].astype(bool)

        # Create DataLoaders using only the respective masks for each set
        train_loader = DataLoader(data[train_mask], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(data[val_mask], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(data[test_mask], batch_size=batch_size, shuffle=False)

        # Run the experiment 10 times and collect metrics
        micro_f1_scores = []
        macro_f1_scores = []
        auc_scores = []

        for _ in range(10):  # Repeat 10 times
            model = BiGCN_individual(data.num_features, args.out_feat, data.num_classes).to(device)
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

            # Train the model with early stopping
            train_with_early_stopping(model, train_loader, val_loader, optimizer, device, epochs, patience=10)

            # Evaluate the model on the test set
            micro_f1, macro_f1, auc = evaluate_test_metrics(model, test_loader, device)
            micro_f1_scores.append(micro_f1)
            macro_f1_scores.append(macro_f1)
            auc_scores.append(auc)

        # Calculate standard deviation of metrics
        print(f"Results for r={r}:")
        print(f"Macro F1 Standard Deviation: {np.mean(macro_f1_scores):.4f}, {np.std(macro_f1_scores):.4f}")
        print(f"Micro F1 Standard Deviation: {np.mean(micro_f1_scores):.4f}, {np.std(micro_f1_scores):.4f}")
        print(f"AUC Standard Deviation: {np.mean(auc_scores):.4f}, {np.std(auc_scores):.4f}")
    #
    #     # Optionally store the results for later
    #     all_micro_f1.append(micro_f1_scores)
    #     all_macro_f1.append(macro_f1_scores)
    #     all_auc.append(auc_scores)

    # # Step 1: Collect all labels (data.y) from the dataset
    # all_labels = []
    # for idx in range(len(data)):
    #     sample = data.get(idx)  # Access each sample
    #     all_labels.append(sample.y.item())  # Extract the label and convert to scalar
    #
    # all_labels = np.array(all_labels)  # Convert to a numpy array for train_test_split
    #
    # # Step 2: Split data based on labels
    # for r in [1, 5]:
    #     mask = train_test_split(all_labels, seed=0,
    #                             train_examples_per_class=r,
    #                             val_size=500, test_size=None)
    #     train_mask = mask['train'].astype(bool)
    #     val_mask = mask['val'].astype(bool)
    #     test_mask = mask['test'].astype(bool)
    #
    #     # Step 3: Create DataLoaders for each split
    #     train_data = [data.get(i) for i in range(len(data)) if train_mask[i]]
    #     val_data = [data.get(i) for i in range(len(data)) if val_mask[i]]
    #     test_data = [data.get(i) for i in range(len(data)) if test_mask[i]]
    #
    #     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #     val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    #     test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    #
    #     # Step 4: Continue with training and evaluation
    #     micro_f1_scores = []
    #     macro_f1_scores = []
    #     auc_scores = []
    #
    #     for _ in range(10):  # Repeat 10 times
    #         model = BiGCN_individual(data.num_features, args.out_feat, len(np.unique(all_labels))).to(device)
    #         optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    #
    #         # Train the model with early stopping
    #         train_with_early_stopping(model, train_loader, val_loader, optimizer, device, epochs, patience=10)
    #
    #         # Evaluate the model on the test set
    #         micro_f1, macro_f1, auc = evaluate_test_metrics(model, test_loader, device)
    #         micro_f1_scores.append(micro_f1)
    #         macro_f1_scores.append(macro_f1)
    #         auc_scores.append(auc)
    #
    #     # Step 5: Calculate metrics
    #     print(f"Results for r={r}:")
    #     print(f"Macro F1 Standard Deviation: {np.mean(macro_f1_scores):.4f}, {np.std(macro_f1_scores):.4f}")
    #     print(f"Micro F1 Standard Deviation: {np.mean(micro_f1_scores):.4f}, {np.std(micro_f1_scores):.4f}")
    #     print(f"AUC Standard Deviation: {np.mean(auc_scores):.4f}, {np.std(auc_scores):.4f}")
