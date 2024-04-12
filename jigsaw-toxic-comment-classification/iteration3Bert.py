# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from bert_model import BertModel
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from plot import plot_roc_curve
from ToxicCommentsDataset import ToxicCommentsDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

if os.listdir("./toxic_comment_roberta"):
    model = BertModel()
    model.load("./toxic_comment_roberta")
    print("!!!!!!!!!Model loaded!!!!!!!!")
else:

    dataset_path = "data/input/train.csv"
    print("!!!!!!!!!Training the model!!!!!!!!")
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Preprocess the data
    df['comment_text'] = df['comment_text'].fillna(" ").str.lower()

    # Split the dataset into training and validation sets

    train_df, val_df = train_test_split(df, test_size=0.1)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Initialize and train the model
    model = BertModel(num_labels=6)
    model.train(train_df, val_df)

    # Save the model
    model.save("./toxic_comment_roberta")

test_dataset_path = "data/input/test.csv"
df_test = pd.read_csv(test_dataset_path)
df_test['comment_text'] = df_test['comment_text'].fillna(" ").str.lower()

# Load the test labels
test_labels_path = "data/input/test_labels.csv"
df_test_labels = pd.read_csv(test_labels_path)

# Merge the test dataset with its labels
df_test = df_test.merge(df_test_labels, on='id')

# Filter out any rows where the labels might be missing or marked as '-1' (if applicable)
df_test = df_test[df_test.toxic != -1]
df_test = df_test.reset_index(drop=True)


test_dataset = ToxicCommentsDataset(df_test, model.tokenizer, max_len=128)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def get_test_predictions_and_labels(distilbert_model_instance, data_loader):
    distilbert_model_instance.model.eval()
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(range(len(data_loader)), desc="Evaluating")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
            }
            labels = batch['labels'].to(device)
            outputs = distilbert_model_instance.model(**inputs)
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
predicted_labels = (probabilities > 0.9).astype(int)
submission_df = pd.DataFrame(predicted_labels, columns =label_names)

# Add the 'id' column from the test DataFrame
submission_df.insert(0, 'id', df_test['id'].values)

# Write the DataFrame to a CSV file
submission_df.to_csv('data/input/predictions.csv', index=False)
# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
print(np.shape(true_labels))
roc_aucs = plot_roc_curve(true_labels, probabilities, label_names)

# Compute the mean column-wise ROC AUC
mean_roc_auc = np.mean(roc_aucs)

print(f"Mean column-wise ROC AUC: {mean_roc_auc}")

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
