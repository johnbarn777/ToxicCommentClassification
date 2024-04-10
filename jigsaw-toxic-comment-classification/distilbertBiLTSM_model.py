# Let's first adapt the DistilBertWithBiLSTM class to include a method for loading a model from a saved state.
# This involves defining a save method that saves the model's state_dict and a load method that can load it.

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast, AdamW
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm

class DistilBertWithBiLSTM(nn.Module):
    def __init__(self, num_labels=6, hidden_dim=768, lstm_layers=1, dropout=0.1):
        super(DistilBertWithBiLSTM, self).__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # DistilBERT Model
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim // 2,  # Because it's bidirectional
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)

        # Dropout and Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask=None):
        # DistilBERT output
        distilbert_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask)
        sequence_output = distilbert_output[0]  # (batch_size, sequence_length, hidden_dim)

        # LSTM output
        lstm_output, (h_n, c_n) = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output[:, -1, :])  # Use the last hidden state

        # Classifier
        logits = self.classifier(lstm_output)

        return logits

    def evaluate(self, dataloader):
        self.eval()  # Set the model to evaluation mode
        total_loss = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self(input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions.double() / (len(dataloader.dataset))
        return avg_loss, accuracy

    from tqdm import tqdm

    def train_model(self, train_dataloader, val_dataloader=None, epochs=1, learning_rate=5e-5):

        self.train()  # Set the model to training mode
        optimizer = AdamW(self.parameters(), lr=learning_rate)
        self.loss_fn = CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            self.train()

            # Add tqdm progress bar
            progress_bar = tqdm(train_dataloader, desc='Epoch {:1d}'.format(epoch + 1), leave=False, disable=False)
            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self(input_ids, attention_mask=attention_mask).to(self.device)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"\nEpoch {epoch + 1}, Train Loss: {avg_train_loss}")

            if val_dataloader is not None:
                avg_val_loss, val_accuracy = self.evaluate(val_dataloader)
                print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}")

        print("Training complete.")
    def save(self, save_dir):
        """
        Saves the model's state dict and the tokenizer to the specified directory.
        """
        torch.save(self.state_dict(), f"./BiLTSM/model_state.bin")

    @classmethod
    def load(cls, model_dir, num_labels=6, hidden_dim=768, lstm_layers=1, dropout=0.1):
        """
        Loads the model from the state dict.
        """
        model = cls(num_labels=num_labels, hidden_dim=hidden_dim, lstm_layers=lstm_layers, dropout=dropout)
        model.load_state_dict(torch.load(f"./BiLTSM/model_state.bin"))
        model.eval()  # Set the model to evaluation mode
        return model

# This code defines methods for saving and loading the model, which will be useful for persisting the model's state across sessions.
# The save method saves the state_dict of the model, which includes all the weights.
# The load class method creates a new instance of the model, loads the weights from a saved state_dict, and returns it.
# Note: The tokenizer is typically saved and loaded separately, as it doesn't have stateful parameters that need to be learned during training.
