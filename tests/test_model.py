"""
Test script for ChordGCPT model
"""
import torch
from models.chord_gcpt import ChordGCPT
from config import Config

def test_model_initialization():
    print("Testing model initialization...")
    print("-" * 50)
    
    vocab_size = 500
    
    model = ChordGCPT(
        vocab_size=vocab_size,
        embedding_dim=Config.embedding_dim,
        num_layers=Config.num_layers,
        num_heads=Config.num_heads,
        feedforward_dim=Config.feedforward_dim,
        dropout=Config.dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    assert total_params > 0, "Model has no parameters"
    assert trainable_params == total_params, "Some parameters are not trainable"
    
    print("Model initialization test passed!\n")
    return model

def test_forward_pass(model):
    print("Testing forward pass...")
    print("-" * 50)
    
    batch_size = 2
    seq_len = 20
    vocab_size = 500
    
    # Create dummy input
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {token_ids.shape}")
    
    # Forward pass
    logits = model(token_ids)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {vocab_size})")
    
    # Verify output shape
    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"Output shape mismatch! Got {logits.shape}, expected ({batch_size}, {seq_len}, {vocab_size})"
    
    # Verify output is not all zeros
    assert not torch.all(logits == 0), "Output is all zeros"
    
    # Verify output has reasonable values
    assert torch.isfinite(logits).all(), "Output contains NaN or Inf values"
    
    print("Forward pass test passed!\n")

def test_causal_masking(model):
    print("Testing causal masking...")
    print("-" * 50)
    
    seq_len = 10
    
    # Create mask
    mask = model.generate_square_subsequent_mask(seq_len)
    
    print(f"Mask shape: {mask.shape}")
    print(f"Expected shape: ({seq_len}, {seq_len})")
    
    # Verify mask shape
    assert mask.shape == (seq_len, seq_len), "Mask shape incorrect"
    
    # Verify causal structure (upper triangle should be -inf)
    # Position i should only see positions <= i
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert mask[i, j] == float('-inf'), f"Position ({i},{j}) should be masked"
            else:
                assert mask[i, j] == 0, f"Position ({i},{j}) should not be masked"
    
    print("Causal mask structure verified")
    print("Causal masking test passed!\n")

def test_generation():
    print("Testing generation capability...")
    print("-" * 50)
    
    vocab_size = 500
    batch_size = 1
    start_seq_len = 5
    max_gen_length = 10
    
    # Create model
    model = ChordGCPT(
        vocab_size=vocab_size,
        embedding_dim=128,  # Smaller for faster testing
        num_layers=2,
        num_heads=4,
        feedforward_dim=512,
        dropout=0.1
    )
    
    # Create starting sequence
    input_ids = torch.randint(0, vocab_size, (batch_size, start_seq_len))
    
    print(f"Starting sequence length: {start_seq_len}")
    print(f"Max generation length: {max_gen_length}")
    
    # Generate
    device = 'cpu'
    model.eval()
    output_ids = model.generate(
        input_ids,
        max_length=max_gen_length,
        temperature=1.0,
        top_k=50,
        device=device
    )
    
    print(f"Generated sequence length: {output_ids.shape[1]}")
    print(f"Generation increased length by: {output_ids.shape[1] - start_seq_len}")
    
    # Verify generation
    assert output_ids.shape[1] > start_seq_len, "Generation did not add any tokens"
    assert output_ids.shape[1] <= start_seq_len + max_gen_length, "Generation exceeded max length"
    
    # Verify all tokens are valid
    assert (output_ids >= 0).all() and (output_ids < vocab_size).all(), "Invalid token IDs generated"
    
    print("Generation test passed!\n")

def test_loss_computation():
    print("Testing loss computation...")
    print("-" * 50)
    
    vocab_size = 500
    batch_size = 2
    seq_len = 20
    
    # Create model
    model = ChordGCPT(
        vocab_size=vocab_size,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        feedforward_dim=512,
        dropout=0.1
    )
    
    # Create dummy data
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Split into input and target (shifted by 1)
    input_ids = token_ids[:, :-1]
    target_ids = token_ids[:, 1:]
    
    # Forward pass
    logits = model(input_ids)
    
    # Compute loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1)
    )
    
    print(f"Loss value: {loss.item():.4f}")
    
    # Verify loss is reasonable
    assert torch.isfinite(loss), "Loss is NaN or Inf"
    assert loss.item() > 0, "Loss should be positive"
    
    # Test backward pass
    loss.backward()
    
    # Verify gradients exist
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_gradients, "No gradients computed"
    
    print("Loss computation and backward pass successful")
    print("Loss computation test passed!\n")

if __name__ == "__main__":
    # Run all tests
    model = test_model_initialization()
    test_forward_pass(model)
    test_causal_masking(model)
    test_generation()
    test_loss_computation()
    

    print("All model tests passed!")
