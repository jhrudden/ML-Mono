from .simple_rnn import SimpleRNN
import torch
from torch import nn

class LanguageModelSRNN(nn.Module):
    """
    Next token prediction model implemented with a simple RNN.
    """
    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, use_embedding: bool = False, embedding_dim: int = None):
        super(LanguageModelSRNN, self).__init__()
        self.rnn = SimpleRNN(vocab_size, hidden_dim, vocab_size)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim if use_embedding else vocab_size

        if use_embedding:
            assert embedding_dim is not None, f'embedding_dim must be specified if use_embedding is True'
            self.embedding_dim = embedding_dim
            self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        
    def embedding(self, X):
        """
        Returns the embedding of the input. If use_embedding is False, returns a one hot encoding of the input.

        :param X: input tensor of shape (batch_size, seq_len)
        :return: embedding tensor of shape (batch_size, seq_len, embedding_dim)
        """
        if self.use_embedding:
            return self.embedding_layer(X)
        else:
            return nn.functional.one_hot(X, num_classes=self.embedding_dim).float()
    
    def forward(self, X):
        """
        Forward pass of the model.
        :param x: input tensor of shape (batch_size, seq_len)
        :return: output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # one hot encode input
        X_embedding = self.embedding(X) # (batch_size, seq_len, vocab_size)
        B, S, V = X_embedding.size()
        outputs = torch.zeros((B, S, V))
        logits, _ = self.rnn(X_embedding)

        # A little sad to have to re loop through the sequence
        for t in range(S):
            h_t = logits[:, t, :]
            logits_t = self.linear(h_t)
            softmaxed = torch.softmax(logits_t, dim=1)
            outputs[:, t, :] = softmaxed

        return outputs
    
    @torch.no_grad()
    def generate(self, context, seq_len: int, num_samples: int = 1):
        """
        Generate a sequence of text using the model.
        :param seq_len: the length of the sequence to generate
        :return: a sequence of text
        """
        self.eval()
        for i in range(num_samples):
            current_context = context[-min(seq_len, len(context)):]
            current_context = self.embedding(current_context.view(1, -1))
            _, h = self.rnn(current_context)
            logits = self.linear(h)
            softmaxed = torch.softmax(logits, dim=1)
            next_char = torch.multinomial(softmaxed, num_samples=1).view(-1)
            context = torch.cat([context, next_char], dim=0)
        resultant_sequence = context.detach().tolist()
        self.train()
        return resultant_sequence