
import torch
from transformers import AutoModelForCausalLM, AutoConfig

class Model(torch.nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=self.config)

    def forward(self, x):
        return self.model(x).logits

model_name = "facebook/opt-1.3b"
config = AutoConfig.from_pretrained(model_name)
vocab_size = config.vocab_size
sequence_length = 256
batch_size = 32

def get_inputs():
    inputs = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return [inputs]

def get_init_inputs():
    return [model_name, config]