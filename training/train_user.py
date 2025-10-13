import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model

from models.chord_gcpt import ChordGCPT
from config import Config

class UserDataset(Dataset):
    def __init__(self, user_songs, vocab):
        """
        user_songs: list of chord progression strings
        Example: ["C Am F G", "C F Dm G"]
        """
        self.vocab = vocab
        self.sequences = []
        
        for song in user_songs:
            tokens = ['<START>'] + song.split() + ['<EOS>']
            token_ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
            self.sequences.append(torch.tensor(token_ids, dtype=torch.long))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_fn(batch):
    """Pad sequences to same length"""
    max_len = max(len(seq) for seq in batch)
    
    padded = []
    for seq in batch:
        padding = torch.full((max_len - len(seq),), 0, dtype=torch.long)
        padded.append(torch.cat([seq, padding]))
    
    return torch.stack(padded)

def train_user_lora(user_songs, user_id, genre="POP"):
    """
    Train user-specific LoRA adapter
    
    user_songs: list of chord progressions
    user_id: unique identifier for user
    genre: optional genre specification
    """
    
    # Load vocab
    with open(f'{Config.model_dir}/vocab.json', 'r') as f:
        vocab = json.load(f)
    
    # Add genre token to user songs
    genre_token = f"<GENRE_{genre.upper()}>"
    user_songs_with_genre = [f"{genre_token} {song}" for song in user_songs]
    
    # Create dataset
    dataset = UserDataset(user_songs_with_genre, vocab)
    dataloader = DataLoader(
        dataset, 
        batch_size=Config.user_batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Load base model
    print("Loading base model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base_model = ChordGCPT(
        vocab_size=len(vocab),
        embedding_dim=Config.embedding_dim,
        num_layers=Config.num_layers,
        num_heads=Config.num_heads,
        feedforward_dim=Config.feedforward_dim,
        dropout=Config.dropout,
        chord_mappings_path=Config.chord_mappings_path
    ).to(device)
    
    base_model.load_state_dict(torch.load(f'{Config.model_dir}/chord_gcpt_final.pt', map_location=device))
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        target_modules=["transformer.layers.*.self_attn.in_proj_weight"],  # Attention layers
        lora_dropout=Config.lora_dropout,
        bias="none",
    )
    
    # Wrap model with LoRA
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.user_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    print(f"Training user {user_id} for {Config.user_epochs} epochs...")
    
    for epoch in range(Config.user_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            
            # Input and target
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            
            # Forward
            logits = model(input_ids)
            
            # Loss
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
        
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{Config.user_epochs} - Loss: {avg_loss:.4f}")
    
    # Save LoRA adapter
    lora_dir = f"{Config.model_dir}/user_{user_id}_lora"
    os.makedirs(lora_dir, exist_ok=True)
    model.save_pretrained(lora_dir)
    print(f"User LoRA saved: {lora_dir}")
    
    return model

if __name__ == "__main__":
    # Example usage
    user_songs = [
        "C Am F G",
        "C F Dm G",
        "C G Am F",
        "C Am Dm G",
        "C F G Am"
    ]
    
    train_user_lora(user_songs, user_id="test_user", genre="POP")