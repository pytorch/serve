from huggingface_hub import snapshot_download


def download_model(
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    revision="main",
    model_path=".cache",
    use_auth_token=True,
):
    # Only download pytorch checkpoint files
    allow_patterns = [
        "*.json",
        "*.pt",
        "*.bin",
        "*.txt",
        "*.model",
        "*.pth",
        "*.safetensors",
        "original/*",
    ]

    snapshot_path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=allow_patterns,
        cache_dir=model_path,
        use_auth_token=use_auth_token,
    )

    return snapshot_path
