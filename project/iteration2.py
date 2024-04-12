# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

# Assuming the Kaggle dataset is downloaded and stored in 'data/' directory
dataset_path = "data/input/train.csv"

# Load the dataset
df = pd.read_csv(dataset_path)
# sampling 10% for faster training and testing
df = df.sample(frac=0.85, random_state=42)
# Preprocess the data
df['comment_text'] = df['comment_text'].fillna(" ").str.lower()

# Split the dataset into training and validation sets

train_df, val_df = train_test_split(df, test_size=0.1)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
# Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Define a custom dataset
class ToxicCommentsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(self.targets[index], dtype=torch.float)
}

# Define the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory for model predictions and checkpoints
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Convert dataframes into torch Dataset
train_dataset = ToxicCommentsDataset(train_df, tokenizer, max_len=128)
val_dataset = ToxicCommentsDataset(val_df, tokenizer, max_len=128)

# Define the Trainer
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Proceed to train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./DistilbertModel")
tokenizer.save_pretrained("./DistilbertModel")

# Note: For actual training and model saving, remove the comments and ensure the necessary libraries and dataset are properly set up.
# The evaluation part, including accuracy and other metrics calculation, is crucial for understanding the model's performance and should be conducted accordingly.


