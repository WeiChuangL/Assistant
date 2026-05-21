# Chat models available via NVIDIA API
CHAT_MODELS = {
    "deepseek-v4-flash": "deepseek-ai/deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-ai/deepseek-v4-pro",
    "glm-5.1": "z-ai/glm-5.1",
    "qwen3-coder": "qwen/qwen3-coder-480b-a35b-instruct",
    "minimax-m2.7": "minimaxai/minimax-m2.7",
    "gemma-4-31b": "google/gemma-4-31b-it",
}

# Embedding models available via NVIDIA API
EMBEDDING_MODELS = {
    "nv-embedqa-e5-v5": "nvidia/nv-embedqa-e5-v5",
    "nv-embedqa-mistral-7b-v2": "nvidia/nv-embedqa-mistral-7b-v2",
    "nv-embedqa-e5-v4": "nvidia/nv-embedqa-e5-v4",
}

# Default embedding dimensions for known models
EMBEDDING_DIMENSIONS = {
    "nvidia/nv-embedqa-e5-v5": 1024,
    "nvidia/nv-embedqa-mistral-7b-v2": 4096,
    "nvidia/nv-embedqa-e5-v4": 1024,
}
