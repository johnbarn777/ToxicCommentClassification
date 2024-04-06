# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if(os.listdir("./toxic_comment_model")):
    model = DistilBertForSequenceClassification.from_pretrained("./toxic_comment_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("./toxic_comment_model")

    model = model.to(device)
else:

     # Assuming the Kaggle dataset is downloaded and stored in 'data/' directory
    dataset_path = "data/input/train.csv"
    print("!!!!!!!!!Training the model!!!!!!!!")
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
    output_dir='./results',
    num_train_epochs=2,  # Reduced number of epochs
    per_device_train_batch_size=16,  # Increased batch size, adjust based on your GPU
    per_device_eval_batch_size=32,  # Increased evaluation batch size
    warmup_steps=300,  # Reduced warmup steps
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,  # Less frequent logging
    fp16=True,  # Use mixed precision training
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
    model.save_pretrained("./toxic_comment_model")
    tokenizer.save_pretrained("./toxic_comment_model")

test_dataset_path = "data/input/test.csv"
df_test = pd.read_csv(test_dataset_path)
df_test['comment_text'] = df_test['comment_text'].fillna(" ").str.lower()

# Load the test labels
test_labels_path = "data/input/test_labels.csv"
df_test_labels = pd.read_csv(test_labels_path)

# Merge the test dataset with its labels
df_test = df_test.merge(df_test_labels, on='id')

# Filter out any rows where the labels might be missing or marked as '-1' (if applicable)
df_test_filtered = df_test[df_test.toxic >= 0]  # Example condition, adjust based on actual label marking for unusable labels
df_test_filtered.reset_index(drop=True, inplace=True)

class ToxicCommentsTestDataset(Dataset):
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

test_dataset = ToxicCommentsTestDataset(df_test_filtered, tokenizer, max_len=128)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def get_test_predictions_and_labels(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(range(len(data_loader)), desc="Evaluating")
    with torch.no_grad():
        for batch in data_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.append(logits.detach().cpu().numpy())
            true_labels.append(labels.cpu().numpy())

            progress_bar.update(1)
    progress_bar.close()

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    return predictions, true_labels

# Get predictions and true labels
predictions, true_labels = get_test_predictions_and_labels(model, test_loader)

# Apply a sigmoid function to the predictions since we're dealing with multi-label classification
sigmoid = torch.nn.Sigmoid()
probabilities = sigmoid(torch.tensor(predictions)).numpy()

# Convert probabilities to binary predictions using a threshold (e.g., 0.5)
predicted_labels = (probabilities > 0.5).astype(int)
submission_df = pd.DataFrame(predicted_labels, columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

# Add the 'id' column from the test DataFrame
submission_df.insert(0, 'id', df_test_filtered['id'].values)

# Write the DataFrame to a CSV file
submission_df.to_csv('data/input/predictions.csv', index=False)
# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
