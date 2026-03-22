from tqdm import tqdm
import torch
import torch.nn.functional as F
import tiktoken
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import format_input, format_response

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # truncate the texts larger than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = 0 if self.data.iloc[index]["Label"] == "ham" else 1
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.encoded_texts)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in tqdm(data):
            instruction_input = format_input(entry) + format_response(entry)
            self.encoded_texts.append(
                torch.tensor(tokenizer.encode(instruction_input), dtype=torch.long)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0, pin_memory=False):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader

def custom_collate(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None):
    batch_tensors = [torch.as_tensor(item, dtype=torch.long) for item in batch]

    padded = pad_sequence(batch_tensors, batch_first=True, padding_value=pad_token_id)
    # Extra column so the longest sequence still has a next-token target
    padded = F.pad(padded, (0, 1), value=pad_token_id)

    if allowed_max_length is not None:
        padded = padded[:, :allowed_max_length + 1]

    inputs = padded[:, :-1]
    targets = padded[:, 1:].clone()

    # Keep the first pad_token per sequence (EOS signal), mask subsequent padding.
    # cross_entropy ignores targets labeled ignore_index (-100) by default.
    pad_cumsum = (targets == pad_token_id).long().cumsum(dim=1)
    targets[pad_cumsum > 1] = ignore_index

    return inputs, targets
