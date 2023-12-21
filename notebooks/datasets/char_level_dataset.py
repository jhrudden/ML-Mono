import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CharLevelDataset(Dataset):
    def __init__(self, lines, pad_token='<PAD>', seq_len=None):
        self.lines = [list(line) for line in lines]
        self.seq_len = seq_len
        self.pad_token = pad_token
        self.vocab = self.build_vocab(lines, pad_token)
        self.pad_index = self.vocab[pad_token]

    def build_vocab(self, lines, pad_token):
        unique_chars = set(''.join(lines))
        vocab = {char: i for i, char in enumerate(sorted(unique_chars), start=1)}  # start=1 to reserve 0 for padding
        vocab[pad_token] = 0  # Pad token
        return vocab

    def line_to_tensor(self, line):
        print("test", line)
        indices = [self.vocab[char] for char in line]
        if self.seq_len is not None:
            padded_indices = indices[:self.seq_len] + [self.pad_index] * max(0, self.seq_len - len(indices))
        else:
            padded_indices = indices
        return torch.tensor(padded_indices, dtype=torch.long)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        input_tensor = self.line_to_tensor(line)
        target_tensor = self.line_to_tensor(line[1:] + [self.pad_token])  # Shift by one for next character prediction
        return input_tensor, target_tensor


