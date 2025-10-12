import torch
from models.embeddings import HybridEmbedding

# Dummy vocab and embedding
vocab_size = 100
embedding_dim = 256

model = HybridEmbedding(vocab_size, embedding_dim)

# Test input
batch_size = 2
seq_len = 10

token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
token_types = torch.randint(0, 2, (batch_size, seq_len))  # 0=structure, 1=chord

# Forward pass
output = model(token_ids, token_types)

# Check output shape
print(f"Input shape: {token_ids.shape}")
print(f"Output shape: {output.shape}")
print(f"Expected: ({batch_size}, {seq_len}, {embedding_dim})")

# Verify
assert output.shape == (batch_size, seq_len, embedding_dim), "Shape mismatch!"
print("Embedding test completed sucessfully")