import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model.

        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the RNN cell components (input to hidden, hidden to hidden, and hidden to output)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # Input to hidden
        self.h2o = nn.Linear(hidden_size, output_size)  # Hidden to output
        self.tanh = nn.Tanh()  # Activation function for hidden state

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN.

        :param x: Input tensor of shape (seq_len, batch_size, input_size)
        :param h0: Initial hidden state tensor of shape (batch_size, hidden_size)
        :return: Output tensor of shape (seq_len, batch_size, output_size)
        """
        seq_len, batch_size, _ = x.size()
        hidden = h0.to(x.device)
        outputs = []

        for t in range(seq_len):
            combined = torch.cat((x[t], hidden), dim=1)  # Concatenate input and hidden state
            hidden = self.tanh(self.i2h(combined))  # Update hidden state
            output = self.h2o(hidden)  # Compute output
            outputs.append(output)

        return torch.stack(outputs, dim=0)  # (seq_len, batch_size, output_size)

# === Test configuration ===
batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    return [
        torch.rand(sequence_length, batch_size, input_size),
        torch.rand(batch_size, hidden_size)
    ]

def get_init_inputs():
    return [input_size, hidden_size, output_size]
