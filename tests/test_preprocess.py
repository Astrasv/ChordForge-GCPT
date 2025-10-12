"""
Test script for data preprocessing functions
"""
import re
from data.preprocess import normalize_tag, handle_alternate

def test_normalize_tag():
    print("Testing normalize_tag function...")
    print("-" * 50)
    
    test_cases = [
        ("<intro_21>", "<intro>"),
        ("<verse_75>", "<verse>"),
        ("<chorus_3>", "<chorus>"),
        ("<bridge_10>", "<bridge>"),
        ("<intro>", "<intro>"),  # Already normalized
        ("<START>", "<START>"),  # Special token
    ]
    
    for input_tag, expected in test_cases:
        result = normalize_tag(input_tag)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status}: {input_tag} -> {result} (expected: {expected})")
        assert result == expected, f"Failed: {input_tag} -> {result}, expected {expected}"
    
    print("All normalize_tag tests passed!\n")

def test_handle_alternate():
    print("Testing handle_alternate function...")
    print("-" * 50)
    
    test_cases = [
        ("A/Cs", "A"),      # Alternate chord
        ("Dm/F", "Dm"),     # Alternate chord
        ("C/E", "C"),     
        ("G/B", "G"),     
        ("C", "C"),         # Regular chord
        ("Fmaj7", "Fmaj7"), # Regular chord with quality
    ]
    
    for input_chord, expected in test_cases:
        result = handle_alternate(input_chord)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status}: {input_chord} -> {result} (expected: {expected})")
        assert result == expected, f"Failed: {input_chord} -> {result}, expected {expected}"
    
    print("All handle_alternate tests passed!\n")

def test_sequence_processing():
    print("Testing sequence processing...")
    print("-" * 50)
    
    # Simulate a chord sequence
    raw_sequence = "<intro_21> C Am F G <verse_75> Dm/F G C"
    tokens = raw_sequence.split()
    
    print(f"Input sequence: {raw_sequence}")
    
    # Process tokens
    processed = []
    for token in tokens:
        if token.startswith('<') and token.endswith('>'):
            token = normalize_tag(token)
        else:
            token = handle_alternate(token)
        processed.append(token)
    
    print(f"Processed tokens: {' '.join(processed)}")
    
    # Verify normalization happened
    assert "<intro>" in processed, "intro tag not normalized"
    assert "<verse>" in processed, "verse tag not normalized"
    assert "<intro_21>" not in processed, "intro_21 should be normalized"
    assert "<verse_75>" not in processed, "verse_75 should be normalized"
    
    # Verify alternate handling
    assert "Dm" in processed, "Alternate chord not handled"
    assert "Dm/F" not in processed, "Dm/F should become Dm"
    
    print("Sequence processing test passed!\n")

def test_chunking_logic():
    print("Testing chunking logic...")
    print("-" * 50)
    
    # Create a long sequence
    long_sequence = ['<GENRE_POP>', '<START>'] + ['C'] * 300 + ['<EOS>']
    max_length = 256
    
    print(f"Long sequence length: {len(long_sequence)}")
    print(f"Max length: {max_length}")
    
    # Chunk it
    chunks = []
    for i in range(0, len(long_sequence), max_length - 2):
        chunk = long_sequence[i:i + max_length]
        if len(chunk) > 2:
            chunks.append(chunk)
    
    print(f"Number of chunks created: {len(chunks)}")
    print(f"Chunk sizes: {[len(c) for c in chunks]}")
    
    # Verify chunking
    assert len(chunks) > 1, "Should create multiple chunks for long sequence"
    assert all(len(c) <= max_length for c in chunks), "Chunks exceed max length"
    
    print("Chunking logic test passed!\n")

if __name__ == "__main__":
    test_normalize_tag()
    test_handle_alternate()
    test_sequence_processing()
    test_chunking_logic()
    
    print("All preprocessing tests passed!")
    