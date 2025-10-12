import re
import json
from datasets import load_dataset
from collections import defaultdict

def normalize_tag(tag):
    """Normalize <intro_21> -> <intro>"""
    if tag.startswith('<') and tag.endswith('>'):
        content = tag[1:-1]
        base = re.split(r'_\d+', content)[0]
        return f'<{base}>'
    return tag

def handle_alternate(chord):
    """Handle A/Cs -> A"""
    if '/' in chord:
        parts = chord.split('/')
        # Simple heuristic: if second part looks like chord, take first
        if len(parts[1]) > 2:
            return parts[0]
    return chord

def preprocess_dataset():
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("ailsntua/Chordonomicon")
    
    # Keep only necessary columns
    ds = ds.remove_columns([col for col in ds['train'].column_names 
                           if col not in ['id', 'chords', 'main_genre']])
    
    print(f"Dataset size: {len(ds['train'])}")
    
    # Build vocabularies
    genre_vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<EOS>': 3}
    structure_vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<EOS>': 3}
    chord_vocab = set()
    
    print("Building vocabularies...")
    for item in ds['train']:
        # Add genre
        genre = f"<GENRE_{item['main_genre'].upper()}>"
        if genre not in genre_vocab:
            genre_vocab[genre] = len(genre_vocab)
        
        # Process chords
        tokens = item['chords'].split()
        for token in tokens:
            if token.startswith('<') and token.endswith('>'):
                # Structure tag
                token = normalize_tag(token)
                if token not in structure_vocab:
                    structure_vocab[token] = len(structure_vocab)
            else:
                # Chord
                token = handle_alternate(token)
                chord_vocab.add(token)
    
    # Merge vocabularies
    full_vocab = {**genre_vocab, **structure_vocab}
    for chord in sorted(chord_vocab):
        full_vocab[chord] = len(full_vocab)
    
    print(f"Vocabulary size: {len(full_vocab)}")
    print(f"  Genres: {len(genre_vocab)}")
    print(f"  Structures: {len(structure_vocab)}")
    print(f"  Chords: {len(chord_vocab)}")
    
    # Process sequences
    processed_data = []
    
    print("Processing sequences...")
    for item in ds['train']:
        genre_token = f"<GENRE_{item['main_genre'].upper()}>"
        tokens = item['chords'].split()
        
        # Preprocess tokens
        processed_tokens = []
        for token in tokens:
            if token.startswith('<') and token.endswith('>'):
                token = normalize_tag(token)
            else:
                token = handle_alternate(token)
            processed_tokens.append(token)
        
        # Add special tokens
        sequence = [genre_token, '<START>'] + processed_tokens + ['<EOS>']
        
        # Chunk if too long
        max_length = 256
        if len(sequence) > max_length:
            for i in range(0, len(sequence), max_length - 2):
                chunk = sequence[i:i + max_length]
                if len(chunk) > 2:  # Must have at least genre, start, and 1 token
                    processed_data.append({
                        'id': f"{item['id']}_chunk_{i}",
                        'tokens': chunk,
                        'genre': item['main_genre']
                    })
        else:
            processed_data.append({
                'id': item['id'],
                'tokens': sequence,
                'genre': item['main_genre']
            })
    
    print(f"Processed {len(processed_data)} sequences")
    
    # Save
    with open('data/vocab.json', 'w') as f:
        json.dump(full_vocab, f, indent=2)
    
    with open('data/processed_data.json', 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print("Preprocessing complete!")
    print("Files saved:")
    print("  - data/vocab.json")
    print("  - data/processed_data.json")
    
    return full_vocab, processed_data

if __name__ == "__main__":
    preprocess_dataset()