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