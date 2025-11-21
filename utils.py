# Standard library
from collections import defaultdict

# Third-party: PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Third-party: HuggingFace Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

def load_model_and_tokenizer(model_name_or_path):
    """
    Generic function to load any causal language model and its tokenizer.
    Handles different model architectures and tokenizer configurations.
    """
    print(f"Loading model: {model_name_or_path} ...")


    if "gemma" in model_name_or_path.lower():
        data_type = torch.bfloat16
    else:
        data_type = torch.float16
    # Load the model with generic parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=data_type,
        device_map="auto",
        attn_implementation="eager",
    ).eval()

    # Load tokenizer with model-specific handling
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side="left", legacy=False
    )

    # Generic padding token handling
    if tokenizer.pad_token is None:
        # Try to use EOS token as pad token (common for many models)
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Using EOS token as pad token: {tokenizer.eos_token}")
        else:
            # Fallback to a default pad token
            tokenizer.pad_token = "<pad>"
            print("Using default pad token: <pad>")

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    print(f"Model: {model.config.model_type}")
    print(f"Tokenizer pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    return model, tokenizer

class TopicRNN(nn.Module):
    def __init__(self, input_dim=1024, num_layers=14, hidden_dim=256, num_rnn_layers=3, num_topics=5, rnn_type="GRU"):
        """
        RNN-based model for topic classification.

        Args:
            projection_dim (int): Size of input embeddings (after projection).
            num_layers (int): Number of layers stacked in LLM representation.
            hidden_dim (int): Hidden size of the RNN.
            num_rnn_layers (int): Number of RNN layers.
            num_topics (int): Number of topic categories.
            rnn_type (str): "GRU" or "LSTM".
        """    
        super(TopicRNN, self).__init__()

        self.input_dim = input_dim * num_layers 
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers
        self.num_topics = num_topics
        self.rnn_type = rnn_type

        # RNN layer (GRU or LSTM)
        self.rnn = nn.GRU(
            input_size=self.input_dim,  
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,  
            dropout=0.3
        )

        # Output layer (Multi-label classification)
        self.fc = nn.Linear(hidden_dim * 2, num_topics)  # *2 because bidirectional

    def forward(self, x):
        """
        Forward pass of the RNN.

        Args:
            x (Tensor): Shape (batch_size, seq_len, /49152)

        Returns:
            output (Tensor): Shape (batch_size, num_topics)
        """
      
        # Ensure input is in float32
        x = x.float()  # Convert from float16 to float32 if needed

        # Forward through GRU
        rnn_out, _ = self.rnn(x)  # Shape: (batch_size, seq_len, hidden_dim*2)

        # Take the last hidden state
        final_hidden_state = rnn_out[:, -1, :]  # Shape: (batch_size, hidden_dim*2)

        # Classifier
        output = self.fc(final_hidden_state)  # Shape: (batch_size, num_topics)

        return output
