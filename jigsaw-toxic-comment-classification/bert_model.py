# Assuming base_model.py and reoberta_model.py are in the same directory
from base_model import BaseModel

from transformers import BertTokenizerFast , BertForSequenceClassification, Trainer, TrainingArguments

import torch
from ToxicCommentsDataset import ToxicCommentsDataset  # Ensure this import statement works based on your project structure

class BertModel(BaseModel):
    def __init__(self, num_labels=6):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
    
    def train(self, train_df, val_df):
        # Initialize the tokenizer and model
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_labels).to(self.device)

        # Training Arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=300,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            fp16=True,
        )

        # Convert dataframes into torch Dataset
        train_dataset = ToxicCommentsDataset(train_df, self.tokenizer, max_len=128)
        val_dataset = ToxicCommentsDataset(val_df, self.tokenizer, max_len=128)

        # Define the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Train the model
        trainer.train()

    def save(self, path="./toxic_comment_bert"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path="./toxic_comment_bert"):
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizerFast.from_pretrained(path)
        self.model = self.model.to(self.device)

    # Implement other required methods...
