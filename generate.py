import torch
import json
from peft import PeftModel

from models.chord_gcpt import ChordGCPT
from config import Config

def generate_progression(prompt, genre="POP", user_id=None, max_length=64):
    """
    Generate chord progression
    
    prompt: starting chords (e.g., "C Am")
    genre: genre specification
    user_id: if specified, load user's LoRA adapter
    max_length: maximum generation length
    """
    
    # Load vocab
    with open(f'{Config.model_dir}/vocab.json', 'r') as f:
        vocab = json.load(f)
    
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Load base model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ChordGCPT(
        vocab_size=len(vocab),
        embedding_dim=Config.embedding_dim,
        num_layers=Config.num_layers,
        num_heads=Config.num_heads,
        feedforward_dim=Config.feedforward_dim,
        dropout=Config.dropout,
        chord_mappings_path=Config.chord_mappings_path
    ).to(device)
    
    model.load_state_dict(torch.load(f'{Config.model_dir}/chord_gcpt_final.pt', map_location=device))
    
    # Load user LoRA if specified
    if user_id:
        print(f"Loading user {user_id} LoRA adapter...")
        lora_dir = f"{Config.model_dir}/user_{user_id}_lora"
        model = PeftModel.from_pretrained(model, lora_dir)
    
    model.eval()
    
    # Prepare input
    genre_token = f"<GENRE_{genre.upper()}>"
    input_tokens = [genre_token, '<START>'] + prompt.split()
    input_ids = [vocab.get(t, vocab['<UNK>']) for t in input_tokens]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Generate
    print(f"Generating from: {' '.join(input_tokens)}")
    
    output_ids = model.generate(
        input_tensor, 
        max_length=max_length,
        temperature=Config.temperature,
        top_k=Config.top_k,
        top_p=Config.top_p,
        device=device
    )
    
    # Decode
    output_tokens = [id_to_token[idx.item()] for idx in output_ids[0]]
    
    # Remove special tokens for display
    clean_output = [t for t in output_tokens if not t.startswith('<')]
    
    print(f"\nGenerated progression:")
    print(' '.join(clean_output))
    
    return clean_output

if __name__ == "__main__":
    # Example 1: Generate with base model
    print("=" * 60)
    print("Example 1: Base model generation (Pop)")
    print("=" * 60)
    generate_progression("C", genre="POP", max_length=32)
    
    print("\n" + "=" * 60)
    print("Example 2: Base model generation (Jazz)")
    print("=" * 60)
    generate_progression("Cmaj7", genre="JAZZ", max_length=32)
    
    # Example 3: User-specific generation (requires trained user LoRA)
    # print("\n" + "=" * 60)
    # print("Example 3: User-specific generation")
    # print("=" * 60)
    # generate_progression("C", genre="POP", user_id="test_user", max_length=32)