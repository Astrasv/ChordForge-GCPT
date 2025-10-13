import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os

from models.chord_gcpt import ChordGCPT
from config import Config

class ChordDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        
        # Convert to IDs
        token_ids = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        
        return torch.tensor(token_ids, dtype=torch.long)

def collate_fn(batch):
    """Pad sequences to same length"""
    max_len = max(len(seq) for seq in batch)
    
    padded = []
    for seq in batch:
        padding = torch.full((max_len - len(seq),), 0, dtype=torch.long)  # <PAD> = 0
        padded.append(torch.cat([seq, padding]))
    
    return torch.stack(padded)

def train():
    # Load data
    print("Loading preprocessed data...")
    with open('data/vocab.json', 'r') as f:
        vocab = json.load(f)
    
    with open('data/processed_data.json', 'r') as f:
        data = json.load(f)
    
    # Create dataset
    dataset = ChordDataset(data, vocab)
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.base_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("Initializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = ChordGCPT(
        vocab_size=len(vocab),
        embedding_dim=Config.embedding_dim,
        num_layers=Config.num_layers,
        num_heads=Config.num_heads,
        feedforward_dim=Config.feedforward_dim,
        dropout=Config.dropout,
        chord_mappings_path=Config.chord_mappings_path
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.base_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Training loop
    print(f"Training for {Config.base_epochs} epochs...")
    
    for epoch in range(Config.base_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.base_epochs}")
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Input and target (shifted by 1)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            
            # Forward
            logits = model(input_ids)
            
            # Calculate loss
            loss = criterion(
                logits.reshape(-1, len(vocab)),
                target_ids.reshape(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs(Config.model_dir, exist_ok=True)
            checkpoint_path = f"{Config.model_dir}/chord_gcpt_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = f"{Config.model_dir}/chord_gcpt_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")
    
    # Save vocab
    vocab_path = f"{Config.model_dir}/vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary saved: {vocab_path}")

if __name__ == "__main__":
    train()