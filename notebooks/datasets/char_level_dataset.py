import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = '<PAD>'

class CharLevelDataset(Dataset):
    def __init__(self, corpus, seq_len=None):
        self.seq_len = seq_len
        # TODO: figure out how to use this pad token
        self.vocab = self.build_vocab(corpus)
        self.data = self.process_data(corpus)
        
    def build_vocab(self, lines):
        unique_chars = set(''.join(lines))
        vocab = {char: i for i, char in enumerate(sorted(unique_chars), start=1)}  # start=1 to reserve 0 for padding
        vocab[PAD_TOKEN] = 0  # Pad token
        return vocab
    
    def process_data(self, corpus):
        return torch.tensor([self.vocab[char] for char in corpus], dtype=torch.long)

    def __len__(self):
        return max(len(self.data) - self.seq_len, 0)

    def __getitem__(self, idx):
        data_tensor = self.data[idx:idx+self.seq_len]
        target_tensor = self.data[idx+1:idx+self.seq_len+1]
        return data_tensor, target_tensor

def create_char_level_dataloader(data, batch_size=1, seq_len=100):
    """
    Returns a torch.DataLoader to randomly sample batches of data
    
    Args:
        data: corpus of text data as a string
        batch_size: int
        seq_len: int
    
    Returns:
        dataloader: torch.DataLoader
        vocab_size: int
    """
    dataset = CharLevelDataset(data, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, len(dataset.vocab) + 1  # +1 for padding token