import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Load the dataset
df = pd.read_csv('data/input/train.csv')

# Select the comment text and the labels
comments = df['comment_text'].values
labels = df[df.columns[2:]].values  # Assuming the labels start from the 3rd column

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(comments, labels, test_size=0.1, random_state=42)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

class ToxicCommentsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ToxicCommentsDataset(train_encodings, train_labels)
val_dataset = ToxicCommentsDataset(val_encodings, val_labels)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)

training_args = TrainingArguments(
    output_dir='./results',               # output directory for model predictions and checkpoints
    num_train_epochs=3,                   # total number of training epochs
    per_device_train_batch_size=8,       # batch size per device during training
    per_device_eval_batch_size=32,        # batch size for evaluation
    warmup_steps=500,                     # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                    # strength of weight decay
    logging_dir='./logs',                 # directory for storing logs
    logging_steps=10,                     # log & evaluate every N steps
    evaluation_strategy="steps",          # evaluate after each logging_steps steps
    eval_steps=10,                        # number of steps to run evaluation for
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

trainer.evaluate()
