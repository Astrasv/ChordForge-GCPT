import torch
import torch.nn as nn
import json
import math

class HybridEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, chord_mappings_path=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Load chord mappings if provided
        self.chord_mappings = {}
        self.chord_to_idx = {}
        if chord_mappings_path:
            try:
                with open(chord_mappings_path, 'r') as f:
                    self.chord_mappings = json.load(f)
                # Create reverse mapping
                for idx, (chord, _) in enumerate(self.chord_mappings.items()):
                    self.chord_to_idx[chord] = idx
            except:
                print("Warning: Could not load chord mappings, using standard embeddings")
        
        # Projection for 12D chords
        self.chord_projection = nn.Linear(12, embedding_dim)
        
        # Standard embedding for structure/genre tokens
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding(5000, embedding_dim))
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, token_ids, token_types):
        """
        token_ids: (batch, seq_len) - token IDs
        token_types: (batch, seq_len) - 0=structure/genre, 1=chord
        """
        batch_size, seq_len = token_ids.shape
        
        # Get base embeddings
        embeddings = self.token_embedding(token_ids)
        
        # Add positional encoding
        embeddings = embeddings + self.pos_encoding[:, :seq_len, :]
        
        # Apply layer norm
        embeddings = self.layer_norm(embeddings)
        
        return embeddings