import torch
from model import TextClassifier
import torch.nn as nn
from data import get_dataloaders_mixed
from omegaconf import OmegaConf
import os
import noise_lib
from tqdm import tqdm
import graph_lib

os.environ["TOKENIZERS_PARALLELISM"] = "false"
config = OmegaConf.load('configs/class_config.yaml')

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextClassifier(cfg.vocab_size, cfg.embedding_dim, cfg.hidden_dim, cfg.num_classes).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)

    # Create data loaders for the training and validation sets
    train_loader, valid_loader= get_dataloaders_mixed(cfg)

    noise = noise_lib.get_noise(cfg).to(device)

    graph = graph_lib.get_graph(cfg, device)
    best_loss = 100
    # Iterate over the training data for the specified number of epochs

    start_step = 0
    latest_path = f"checkpoints/latest.pt"

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f"Resuming from epoch {start_epoch}")

    for step in range(start_step, cfg.training.num_training_steps):
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        
        batch = next(train_loader)
        optimizer.zero_grad()
        tokens = batch['input_ids']
        tokens = torch.LongTensor(tokens).to(device)
        ##NOISE TOKENS
        sampling_eps = 1e-3
        t = (1 - sampling_eps) * torch.rand(tokens.shape[0], device=device) + sampling_eps

        sigma, dsigma = noise(t)
        tokens_noisy = graph.sample_transition(tokens, sigma[:, None])


        targets = batch['source']
        targets = torch.LongTensor(targets).to(device)

        outputs = model(tokens_noisy, t)
        loss = criterion(outputs.view(-1, 2), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(tokens)
        total_samples += len(tokens)
        if(step % 50 == 0):
            print('Step: ' + str(step) + ': Loss ' + str(total_loss/total_samples))
        #Save Checkpoint
        if(step % cfg.eval.checkpoint_every == 0):
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, f"checkpoints/checkpoint_steps_{step}.pt")
            # Evaluate on the validation set after every epoch
        


        if(step % cfg.eval.every == 0):

            model.eval()
            total_val_loss = 0.0
            total_val_samples = 0
            with torch.no_grad():
                for eval_step in range(cfg.eval.steps):
                    batch = next(valid_loader)

                    tokens = batch["input_ids"]
                    tokens = torch.LongTensor(tokens).to(device)
                    ##NOISE TOKENS
                    sampling_eps = 1e-3
                    t = (1 - sampling_eps) * torch.rand(tokens.shape[0], device=device) + sampling_eps

                    sigma, dsigma = noise(t)

                    tokens_noisy = graph.sample_transition(tokens, sigma[:, None])
                    targets = batch["source"]
                    targets = torch.LongTensor(targets).to(device)
                    outputs = model(tokens_noisy, t)
                    val_loss = criterion(outputs.view(-1, 2), targets.view(-1))

                    total_val_loss += val_loss.item() * len(tokens)
                    total_val_samples += len(tokens)

            avg_loss = total_loss / total_samples
            avg_val_loss = total_val_loss / total_val_samples
            
            if(step % cfg.eval.checkpoint_every):
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(model.state_dict(), f"checkpoints/best.pt")

                print(f"Steps {step+1}/{cfg.training.num_training_steps}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

if __name__ == '__main__':
    print('good')
    train(config)
