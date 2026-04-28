import torch
import torch.nn as nn
from connectivity import build_connectivity

# These are largely taken from the codebase at https://github.com/Helena-Yuhan-Liu/BioRNN_RichLazy/blob/master/main.py for the main paper, 
# which adapted the code from https://github.com/gyyang/nn-brain/blob/master/RNN%2BDynamicalSystemAnalysis.ipynb. 

class CTRNN(nn.Module):
    """Continuous-time RNN.

    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons

    Inputs:
        input: (seq_len, batch, input_size), network input
        hidden: (batch, hidden_size), initial hidden activity
    """

    def __init__(self, input_size, hidden_size, config):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = config.get("tau", 100.0)
        dt = config.get("dt", None)
        if dt is None:
            self.alpha = 1
        else:
            self.alpha = dt / self.tau
        self.oneminusalpha = 1 - self.alpha

        self.input2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        W = build_connectivity(
            config["connectivity"],
            (hidden_size, hidden_size)
        )
        self.h2h.weight.data = W

    def recurrence(self, input, hidden):
        pre_activation = self.input2h(input) + self.h2h(hidden) 
        h_new = torch.relu(hidden * self.oneminusalpha +
                           pre_activation * self.alpha)
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if hidden is None:
            hidden = torch.zeros(input.shape[1], self.hidden_size, device=input.device)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, hidden

class Net(nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """
    def __init__(self, config):
        super().__init__()
        self.rnn = CTRNN(config["input_dim"], config["hidden_dim"], config)
        self.fc = nn.Linear(config["hidden_dim"], config["output_dim"], bias=False)

    def forward(self, x, hidden=None):
        rnn_activity, hidden = self.rnn(x, hidden)
        out = self.fc(rnn_activity)
        return out, rnn_activity, hidden