import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AdamW
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import accuracy_score
import os
import json
from evaluate import train_test_split
from torch_geometric import seed_everything
from load_data import read_json_file
seed_everything(0)


class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    preds, true_labels = [], []
    probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    mi_f1 = f1_score(true_labels, preds, average="micro")
    ma_f1 = f1_score(true_labels, preds, average="macro")
    auc = roc_auc_score(true_labels, probs)

    return {"mi_f1": mi_f1, "ma_f1": ma_f1, "auc": auc}


def _extract_text(news_path):
    """
    Extract the concatenated text from the source tweet and reactions.

    Args:
        news_path (str): Path to the news directory.

    Returns:
        str: Concatenated text of the source tweet and reactions.
    """
    source_tweets_path = os.path.join(news_path, 'source-tweets', f'{os.path.basename(news_path)}.json')
    reactions_path = os.path.join(news_path, 'reactions')

    try:
        # Extract source tweet
        source_text = ""
        if os.path.exists(source_tweets_path):
            source_data = read_json_file(source_tweets_path)
            if source_data and "text" in source_data:
                source_text = source_data["text"]

        # Extract reactions
        reaction_texts = []
        if os.path.exists(reactions_path):
            for reaction in os.listdir(reactions_path):
                if reaction.startswith('._') or reaction == '.DS_Store':
                    continue
                reaction_path = os.path.join(reactions_path, reaction)
                reaction_data = read_json_file(reaction_path)
                if reaction_data and "text" in reaction_data:
                    reaction_texts.append(reaction_data["text"] if reaction_data["text"] else "Relay")

        # Concatenate all text
        all_text = source_text + " " + " ".join(reaction_texts)
        return all_text.strip()
    except Exception as e:
        print(f"Error processing {news_path}: {e}")
        return None

def extract_texts_and_labels(root):
    """
    Extract concatenated texts and labels from the dataset.

    Args:
        root (str): Path to the root directory of the dataset.

    Returns:
        tuple: A tuple containing two lists:
            - texts: List of concatenated texts for each propagation graph.
            - labels: List of labels corresponding to the texts (0 for non-rumor, 1 for rumor).
    """
    texts = []
    labels = []

    event_list = [
        'germanwings-crash-all-rnr-threads', 'charliehebdo-all-rnr-threads',
        'sydneysiege-all-rnr-threads', 'ebola-essien-all-rnr-threads',
        'gurlitt-all-rnr-threads', 'putinmissing-all-rnr-threads',
        'ferguson-all-rnr-threads', 'ottawashooting-all-rnr-threads',
        'prince-toronto-all-rnr-threads'
    ]

    for event in event_list:
        event_path = os.path.join(root, event)
        if not os.path.exists(event_path):
            continue

        # Process non-rumor data
        non_rumor_path = os.path.join(event_path, 'non-rumours')
        if os.path.exists(non_rumor_path):
            for news in os.listdir(non_rumor_path):
                if news.startswith('._') or news == '.DS_Store':
                    continue
                text = _extract_text(os.path.join(non_rumor_path, news))
                if text:
                    texts.append(text)
                    labels.append(0)  # Label 0 for non-rumor

        # Process rumor data
        rumor_path = os.path.join(event_path, 'rumours')
        if os.path.exists(rumor_path):
            for news in os.listdir(rumor_path):
                if news.startswith('._') or news == '.DS_Store':
                    continue
                text = _extract_text(os.path.join(rumor_path, news))
                if text:
                    texts.append(text)
                    labels.append(1)  # Label 1 for rumor

    return texts, labels

# text = []
# y = []
# raw_dir = "./Data/DRWeiboV3/raw"
# for tweet in os.listdir(raw_dir):
#     post = json.load(open(os.path.join(raw_dir, tweet), "r", encoding="utf-8"))
#
#     root_content = post["source"]["content"]
#     label = post["source"]["label"]
#     y.append(label)
#     combined_text = root_content
#
#     for comment in post["comment"]:
#         comment_content = comment["content"]
#         if comment_content == "":
#             comment_content = "转发"
#         combined_text += " " + comment_content
#
#     text.append(combined_text)

text, y = extract_texts_and_labels(root="./Data/pheme/raw/all-rnr-annotated-threads")
# 假设你已经准备好了 text 和 y 数据以及 FakeNewsDataset 类
for r in [1, 5]:
    results = []
    for _ in range(10):
        mask = train_test_split(np.array(y), seed=0, train_examples_per_class=r, val_size=500, test_size=None)
        train_mask = mask["train"].astype(bool)
        val_mask = mask["val"].astype(bool)
        test_mask = mask["test"].astype(bool)

        train_data = [text[i] for i in range(len(text)) if train_mask[i]]
        val_data = [text[i] for i in range(len(text)) if val_mask[i]]
        test_data = [text[i] for i in range(len(text)) if test_mask[i]]

        train_labels = torch.LongTensor([y[i] for i in range(len(y)) if train_mask[i]])
        val_labels = torch.LongTensor([y[i] for i in range(len(y)) if val_mask[i]])
        test_labels = torch.LongTensor([y[i] for i in range(len(y)) if test_mask[i]])

        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base", cache_dir="./")
        train_dataset = FakeNewsDataset(texts=train_data, labels=train_labels, tokenizer=tokenizer, max_len=512)
        val_dataset = FakeNewsDataset(texts=val_data, labels=val_labels, tokenizer=tokenizer, max_len=512)
        test_dataset = FakeNewsDataset(texts=test_data, labels=test_labels, tokenizer=tokenizer, max_len=512)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2, cache_dir="./")
        optimizer = AdamW(model.parameters(), lr=2e-5)

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model.to(device)

        epochs = 100
        for epoch in range(epochs):
            train_loss = train(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)

        test_metrics = evaluate(model, test_loader, device)
        results.append(test_metrics)

    mi_f1_avg = np.mean([res["mi_f1"] for res in results])
    mi_f1_std = np.std([res["mi_f1"] for res in results])

    ma_f1_avg = np.mean([res["ma_f1"] for res in results])
    ma_f1_std = np.std([res["ma_f1"] for res in results])

    auc_avg = np.mean([res["auc"] for res in results])
    auc_std = np.std([res["auc"] for res in results])

    print(f"Results for {r}-shot:")
    print(f"Mi-F1: {mi_f1_avg:.4f} ± {mi_f1_std:.4f}")
    print(f"Ma-F1: {ma_f1_avg:.4f} ± {ma_f1_std:.4f}")
    print(f"AUC: {auc_avg:.4f} ± {auc_std:.4f}")
