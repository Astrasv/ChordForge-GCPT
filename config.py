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
    
    # Training - Base
    base_epochs = 50
    base_batch_size = 8
    base_lr = 1e-3
    
    # Training - User
    user_epochs = 50
    user_batch_size = 4
    user_lr = 1e-4
    
    # LoRA
    lora_r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    
    # Generation
    temperature = 0.9
    top_k = 40
    top_p = 0.9
    max_gen_length = 64
    
    # Paths
    data_dir = "data"
    model_dir = "checkpoints"
    chord_mappings_path = "data/chord_mappings.json"