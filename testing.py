import torch
from model import TextClassifier
import torch.nn as nn
from data import get_dataset_mixed
from omegaconf import OmegaConf
import os
import noise_lib
from tqdm import tqdm
import graph_lib


###Load Checkpoint###
os.environ["TOKENIZERS_PARALLELISM"] = "false"
cfg = OmegaConf.load('configs/class_config.yaml')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextClassifier(cfg.vocab_size, cfg.embedding_dim, cfg.hidden_dim, cfg.num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)

# Create data loaders for the training and validation sets
train_datasets, valid_datasets= get_dataset_mixed(cfg)

noise = noise_lib.get_noise(cfg).to(device)

graph = graph_lib.get_graph(cfg, device)
# Iterate over the training data for the specified number of epochs

start_step = 0
latest_path = f"checkpoints/latest.pt"


if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1

tokens = train_datasets["input_ids"]
tokens = torch.LongTensor(tokens).to(device)
sources = train_datasets["source"]

for i in range(5):
    print("Model Guess")
    print(model(tokens[i]))
    print("Model Correct")
    print(sources[i])
