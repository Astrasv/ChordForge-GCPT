class Config:
    # Data
    max_length = 256
    chunk_size = 256
    
    # Model
    embedding_dim = 256
    num_layers = 6
    num_heads = 8
    feedforward_dim = 1024
    dropout = 0.1
    
    
    # Paths
    data_dir = "data"
    model_dir = "checkpoints"
    chord_mappings_path = "data/chord_mappings.json"