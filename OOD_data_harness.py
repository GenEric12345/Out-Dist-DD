
import torch
from model import TextClassifier
import torch.nn as nn
from data import get_dataset_mixed, get_dataset
from omegaconf import OmegaConf
import os
import noise_lib
from tqdm import tqdm
import graph_lib
from transformers import GPT2TokenizerFast

import random, numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
###Load Checkpoint###
os.environ["TOKENIZERS_PARALLELISM"] = "false"
cfg = OmegaConf.load('configs/class_config.yaml')

@torch.no_grad()
def noise_determine(batch, classifier, device, noise, graph):
    tokens = batch["input_ids"]
    source = batch["source"]
    min_noise = torch.zeros(tokens.shape[0], device = device)
    mask = source != 0

    OOD_tokens = tokens[mask]
    num_OOD = OOD_tokens.shape[0]

    ts = torch.linspace(0,1,20).to(device)
    found = torch.zeros(num_OOD, dtype = torch.bool, device =device)
    found_t = torch.ones(num_OOD, device = device)


    for t_level in ts:
        t = torch.full((num_OOD,), t_level)
        sigma, dsigma = noise(t)
        OOD_noisy = graph.sample_transition(OOD_tokens, sigma[:, None])
        logits = classifier(OOD_noisy, )
        probs = torch.softmax(logits, dim=-1)[:, 0]
        uncertain = (probs >= 0.4) & (probs <= 0.6)

        new = ~found & uncertain
        found_t[new] = t_level
        found[new] = True


    min_noise[mask] = found_t
    return min_noise





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TextClassifier(cfg.vocab_size, cfg.embedding_dim, cfg.hidden_dim, cfg.num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)

# Create data loaders for the training and validation sets
train_datasets = get_dataset_mixed(cfg.data.train1, cfg.data.train2, "train", cache_dir = cfg.data.cache_dir, block_size=cfg.model.length)

print(len(get_dataset(cfg.data.train1, "train",  cache_dir = cfg.data.cache_dir, block_size=cfg.model.length)))

print(len(get_dataset(cfg.data.train2, "train",  cache_dir = cfg.data.cache_dir, block_size=cfg.model.length)))


print(len(train_datasets))
noise = noise_lib.get_noise(cfg).to(device)

graph = graph_lib.get_graph(cfg, device)
# Iterate over the training data for the specified number of epochs

start_step = 0
latest_path = f"checkpoints/checkpoint_steps_2000.pt"


if os.path.exists(latest_path):
    print("exists")
    checkpoint = torch.load(latest_path)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step'] + 1

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

tokens = train_datasets["input_ids"]
tokens = torch.LongTensor(tokens).to(device)
sources = train_datasets["source"]
model.eval()
with torch.no_grad():
        for i in range(5):

            t = torch.tensor(0.7).unsqueeze(0).to(device)
            sigma, dsigma = noise(t)
            noisy = graph.sample_transition(tokens[i].unsqueeze(0), sigma[:, None])

            print("Model Guess")
            print(model(noisy,t).squeeze(0))
            print("Oracle Correct")
            print(sources[i])
            #print(tokenizer.decode(noisy.squeeze(0)))
