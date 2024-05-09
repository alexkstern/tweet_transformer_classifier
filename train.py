import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import numpy as np
from dataset import TweetDataset, collate_fn
from model import TweetClassifier


def setup_data_loaders(dataset, batch_size, val_split=0.1, shuffle=True, seed=42):
    # Create indices for the split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PyTorch data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, collate_fn=collate_fn)

    return train_loader, validation_loader

def train_model(model, train_loader, validation_loader, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for tweets, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            tweets, targets = tweets.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, loss = model(tweets, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}')
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for tweets, targets in tqdm(validation_loader, desc=f"Epoch {epoch+1} Validation"):
                tweets, targets = tweets.to(device), targets.to(device)
                logits, loss = model(tweets, targets)
                total_val_loss += loss.item()

        print(f'Epoch {epoch+1}, Validation Loss: {total_val_loss/len(validation_loader)}')

                # Validation phase
        model.eval()
        total_val_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for tweets, targets in tqdm(validation_loader, desc=f"Epoch {epoch+1} Validation"):
                tweets, targets = tweets.to(device), targets.to(device)
                logits, loss = model(tweets, targets)
                total_val_loss += loss.item()
                
                # Calculate accuracy
                accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
                total_accuracy += accuracy

        avg_val_loss = total_val_loss / len(validation_loader)
        avg_accuracy = total_accuracy / len(validation_loader)
        print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_accuracy:.2f}')


# Parameters and Data Setup
dataset_path = "./data/train.csv"
dataset = TweetDataset(dataset_path)
batch_size = 4
train_loader, validation_loader = setup_data_loaders(dataset, batch_size)

# Model Setup
vocab_size = dataset.vocab_size
n_embd = 128
padding_length = dataset.padding_length
num_heads = 8
n_layer = 4
num_classes = 2
dropout = 0.1

model = TweetClassifier(vocab_size, n_embd, padding_length, num_heads, n_layer, dropout, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Start Training
train_model(model, train_loader, validation_loader, optimizer, num_epochs=10)
