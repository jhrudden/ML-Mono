# LSTM implementation in PyTorch
# Reference (conceptual):
# Jurafsky, D., & Martin, J.H. (2022). "RNNs and LSTMs." In Speech and Language Processing (3rd ed.). 
# Retrieved from https://web.stanford.edu/~jurafsky/slp3/

from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        :param input_dim: input dimensionality
        :param hidden_dim: hidden dimensionality
        :param output_dim: output dimensionality
        """
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        #########
        # WEIGHTS
        #########

        # Forget gate weights
        self.U_f = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.W_f = nn.Parameter(torch.zeros(hidden_dim, input_dim))

        # Input gate weights / Information gate weights
        self.U_i = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.W_i = nn.Parameter(torch.zeros(hidden_dim, input_dim))

        # Add gate weights
        self.U_a = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.W_a = nn.Parameter(torch.zeros(hidden_dim, input_dim))

        # Output gate weights
        self.U_o = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.W_o = nn.Parameter(torch.zeros(hidden_dim, input_dim))

        # Init weights
        torch.nn.init.xavier_uniform_(self.U_f)
        torch.nn.init.xavier_uniform_(self.W_f)
        torch.nn.init.xavier_uniform_(self.U_i)
        torch.nn.init.xavier_uniform_(self.W_i)
        torch.nn.init.xavier_uniform_(self.U_a)
        torch.nn.init.xavier_uniform_(self.W_a)
        torch.nn.init.xavier_uniform_(self.U_o)
        torch.nn.init.xavier_uniform_(self.W_o)

    def step(self, X, h: torch.tensor = None, c: torch.tensor = None):
        """
        Single step in time for model using LSTM gates.
        :param X: input data tensor of shape (batch_size, input_dim)
        :param h: hidden state representing previous timestep.
                  tensor of shape (batch_size, hidden_dim)
        :param c: cell state representing remembered information collected from previous timestep.
                  tensor of shape (batch_size, hidden_dim)
        """
        # TODO: verify dimensionnality of X, h, c

        if h is None:
            h = torch.zeros((X.size(0), self.hidden_dim))
        else:
            pass # TODO: verify dimensionnality of h

        if c is None:
            c = torch.zeros((X.size(0), self.hidden_dim))
        else:
            pass # TODO: verify dimensionnality of c

        # Forget gate computations
        f_t_input = torch.matmul(self.U_f, h.T) + torch.matmul(self.W_f, X.T)
        f_t = torch.sigmoid(f_t_input) # sigmoid acts as a mask as pushes values to 0 or 1
        k_t = c * f_t # forget gate mask on previous cell state

        # Input gate computations
        g_t_input = torch.matmul(self.U_i, h.T) + torch.matmul(self.W_i, X.T)
        g_t = torch.tanh(g_t_input)

        # Add gate computations
        i_t_input = torch.matmul(self.U_a, h.T) + torch.matmul(self.W_a, X.T)
        i_t = torch.sigmoid(i_t_input) # sigmoid acts as a mask as pushes values to 0 or 1
        j_t = g_t * i_t # select information from g_t to add to cell state via add gate mask

        # New cell state (prev cell state masked by forget gate + contribution of current input masked by add gate)
        c_t = k_t + j_t

        # Output gate computations
        o_t_input = torch.matmul(self.U_o, h.T) + torch.matmul(self.W_o, X.T)
        o_t = torch.sigmoid(o_t_input)

        # New hidden state
        h_t = torch.tanh(c_t) * o_t

        return h_t, c_t



