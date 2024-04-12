import re
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_CKPT = 'distilbert-base-uncased'

# Hyperparameters
MAX_LEN = 320
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = TRAIN_BATCH_SIZE * 2
EPOCHS = 2
LEARNING_RATE = 1e-05
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)

train_data = pd.read_csv('data/input/train.csv.zip')
print("Num. samples:", len(train_data))

label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_data['labels'] = train_data[label_columns].apply(lambda x: list(x), axis=1)

train_data.drop(['id'], inplace=True, axis=1)
train_data.drop(label_columns, inplace=True, axis=1)

def clean_text(txt):
    """Perform some basic cleaning of the text."""
    return re.sub("[^A-Za-z0-9.,;:!?]+", ' ', str(txt))

class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, new_data=False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.comment_text
        self.new_data = new_data
        self.max_len = max_len
        
        if not new_data:
            self.targets = self.data.labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        text = clean_text(text)

        inputs = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length' if self.new_data else False,
            max_length=self.max_len, 
            return_tensors="pt"
        )
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        
        if not self.new_data:
            labels = torch.tensor(self.targets[index], dtype=torch.float)
            return inputs, labels

        return inputs
    
train_size = 0.4

train_df = train_data.sample(frac=train_size, random_state=123)
val_df = train_data.drop(train_df.index).reset_index(drop=True)
val_df = val_df.sample(frac=train_size, random_state=123)
train_df = train_df.reset_index(drop=True)

print("Orig Dataset: {}".format(train_data.shape))
print("Training Dataset: {}".format(train_df.shape))
print("Validation Dataset: {}".format(val_df.shape))

tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, do_lower_case=True)

train_set = MultiLabelDataset(train_df, tokenizer, MAX_LEN)
val_set = MultiLabelDataset(val_df, tokenizer, MAX_LEN)

def dynamic_collate(data):
    """Custom data collator for dynamic padding."""
    inputs = [d for d,l in data]
    labels = torch.stack([l for d,l in data], dim=0)
    inputs = tokenizer.pad(inputs, return_tensors='pt')
    return inputs, labels

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 2, 
                'collate_fn': dynamic_collate}

val_params = {'batch_size': VALID_BATCH_SIZE,
              'shuffle': False,
              'num_workers': 2, 
              'collate_fn': dynamic_collate}

train_loader = DataLoader(train_set, **train_params)
val_loader = DataLoader(val_set, **val_params)

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_CKPT)
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 6)
        )

    def forward(self, inputs):
        bert_output = self.bert(**inputs)
        hidden_state = bert_output.last_hidden_state
        pooled_out = hidden_state[:, 0]
        logits = self.classifier(pooled_out)
        return logits

model = TransformerModel()
model.to(DEVICE);
trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(f"Trainable params: {round(trainable_params/1e6, 1)} M")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_func = nn.BCEWithLogitsLoss()
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    """An accuracy metric for multi-label problems."""
    if sigmoid: 
        inp = inp.sigmoid()
    return ((inp > thresh) == targ.bool()).float().mean()

def train_one_epoch(train_loader, model, loss_func, optimizer, progress_bar=None):
    """Train model over one epoch."""
    model.train()
    size = len(train_loader.dataset)  # Train set size
    
    for i, (data, targets) in enumerate(train_loader):
        # Put inputs and target on DEVICE
        data = {k: v.to(DEVICE) for k, v in data.items()}
        targets = targets.to(DEVICE)
        
        outputs = model(data)
        loss = loss_func(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if progress_bar is not None:
            progress_bar.update(1)
        
        if i % 1000 == 0:
            loss, step = loss.item(), i * len(targets)
            print(f"Loss: {loss:>4f}  [{step:>6d}/{size:>6d}]")
        elif i == len(train_loader) - 1:
            loss = loss.item()
            print(f"Loss: {loss:>4f}  [{size:>6d}/{size:>6d}]")

def validate_one_epoch(val_loader, model, loss_func):
    """Validate model over one epoch."""
    model.eval()
    num_batches = len(val_loader)
    
    valid_loss, acc_multi = 0, 0

    with torch.no_grad():
        for _, (data, targets) in enumerate(val_loader):
            data = {k: v.to(DEVICE) for k, v in data.items()}
            targets = targets.to(DEVICE)

            outputs = model(data)
            valid_loss += loss_func(outputs, targets).item()
            acc_multi += accuracy_multi(outputs, targets)

    valid_loss /= num_batches  # Avg. loss
    acc_multi /= num_batches   # Avg. acc. multi
    print(f"Avg. valid. loss: {valid_loss:>4f}, Acc. multi: {acc_multi:>4f}\n")
num_train_steps = EPOCHS * len(train_loader)
progress_bar = tqdm(range(num_train_steps))

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1} (lr = {lr_sched.get_last_lr()[0]:.2e})\n-------------------------------")
    train_one_epoch(train_loader, model, loss_func, optimizer, progress_bar)
    #if not FOR_SUBMISSION:
    validate_one_epoch(val_loader, model, loss_func)
    lr_sched.step()
