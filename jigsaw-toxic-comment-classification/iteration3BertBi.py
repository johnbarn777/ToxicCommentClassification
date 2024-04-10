import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, accuracy_score, f1_score, recall_score
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
from distilbertBiLTSM_model import DistilBertWithBiLSTM
from ToxicCommentsDataset import ToxicCommentsDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

model = DistilBertWithBiLSTM(num_labels=len(label_names))
model.to(device)
if os.path.exists("./BiLTSM/model_state.bin"):
    model.load_state_dict(torch.load("./BiLTSM/model_state.bin"))
    model.eval()
    print("Model loaded.")
else:
    print("No pre-trained model found. Training will start from scratch.")
    # Load and preprocess dataset
    dataset_path = "data/input/train.csv"
    df = pd.read_csv(dataset_path)

    # Preprocess the dataset (e.g., lowercase conversion, null handling)
    df['comment_text'] = df['comment_text'].fillna(" ").str.lower()

    # Split dataset
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df = train_df.reset_index(drop=True)


    train_dataset = ToxicCommentsDataset(train_df, model.tokenizer, max_len=128)
    val_dataset = ToxicCommentsDataset(val_df, model.tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model.train_model(train_loader, val_loader, epochs=4, learning_rate=5e-5)
    model.save("./BiLTSM")


test_dataset_path = "data/input/test.csv"
df_test = pd.read_csv(test_dataset_path)
df_test['comment_text'] = df_test['comment_text'].fillna(" ").str.lower()

# Load the test labels
test_labels_path = "data/input/test_labels.csv"
df_test_labels = pd.read_csv(test_labels_path)

# Merge the test dataset with its labels
df_test = df_test.merge(df_test_labels, on='id')

# Filter out any rows where the labels might be missing or marked as '-1' (if applicable)
df_test = df_test.sample(frac=0.1, random_state=23)  # Testing on a smaller test base for plots and output
df_test = df_test[df_test.toxic != -1]
df_test = df_test.reset_index(drop=True)

test_dataset = ToxicCommentsDataset(df_test, model.tokenizer, max_len=128)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model.eval()

# Initialize lists to store predictions and true labels
all_preds = []
all_true_labels = []

# Loop over the test dataset
for batch in tqdm(test_loader, desc="Evaluating"):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)  # True labels

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs, dim=1)  # Convert logits to probabilities
        preds = torch.argmax(probs, dim=1)  # Convert probabilities to predicted labels

    # Move predictions and true labels to CPU and convert them to numpy for sklearn compatibility
    all_preds.extend(preds.cpu().numpy())
    all_true_labels.extend(labels.cpu().numpy())

# Convert the collected predictions and true labels into a compatible format for sklearn's classification report
all_preds = np.array(all_preds)
all_true_labels = np.array(all_true_labels)

# Assuming binary classification for simplicity, adjust according to your specific use case
# Flatten the true labels to match the prediction shape
all_true_labels_flattened = all_true_labels.argmax(axis=1)

# Generate the classification report
print(all_preds.shape)
print(all_true_labels.shape)

