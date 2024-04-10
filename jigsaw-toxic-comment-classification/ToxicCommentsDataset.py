import re
from torch.utils.data import Dataset, DataLoader
import torch

class ToxicCommentsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def clean_text(self, text):
        """Custom text cleaning function."""
        # Replace URLs
        text = re.sub(r'http\S+', '[URL]', text)
        # Replace usernames and timestamps
        text = re.sub(r'\(talk\)|\d{2}:\d{2}, \w+ \d{1,2}, \d{4} \(UTC\)', '[META]', text)
        # Normalize new lines and excessive white spaces
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(' +', ' ', text)
        return text

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        # Clean the text with the custom function
        comment_text = self.clean_text(comment_text)

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
