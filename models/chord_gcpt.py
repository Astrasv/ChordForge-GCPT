import torch
import torch.nn as nn
from models.embeddings import HybridEmbedding

class ChordGCPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, 
                 feedforward_dim, dropout, chord_mappings_path=None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Embedding layer
        self.embeddings = HybridEmbedding(vocab_size, embedding_dim, chord_mappings_path)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output head
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, token_ids, token_types=None):
        """
        token_ids: (batch, seq_len)
        token_types: (batch, seq_len) - optional, 0=structure/genre, 1=chord
        """
        batch_size, seq_len = token_ids.shape
        
        # Create token types if not provided (all as structure)
        if token_types is None:
            token_types = torch.zeros_like(token_ids)
        
        # Get embeddings
        embeddings = self.embeddings(token_ids, token_types)
        embeddings = self.dropout(embeddings)
        
        # Create causal mask
        mask = self.generate_square_subsequent_mask(seq_len).to(embeddings.device)
        
        # Pass through transformer
        # Since we use decoder-only, we use memory=embeddings 
        output = self.transformer(embeddings, embeddings, tgt_mask=mask)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def generate(self, input_ids, max_length=64, temperature=0.9, top_k=40, top_p=0.9, device='cuda'):
        """Generate chord progression"""
        self.eval()
        
        current_ids = input_ids.to(device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(current_ids)
                
                # Get next token logits
                next_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[:, indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Check for EOS
                if next_token.item() == 3:  # <EOS> token ID
                    break
        
        return current_ids