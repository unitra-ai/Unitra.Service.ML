"""Download and cache MADLAD-400-3B model to Modal persistent volume.

This script downloads the MADLAD-400-3B machine translation model
from HuggingFace Hub and saves it to a Modal persistent volume
for faster cold starts.

Usage:
    modal run scripts/download_model.py
"""

import os
import time

import modal

# Modal app configuration
app = modal.App("unitra-model-download")

# Use the same volume as the MT service
volume = modal.Volume.from_name("unitra-models", create_if_missing=True)

# Image with minimal dependencies for download
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "sentencepiece",
        "huggingface_hub",
        "safetensors",
    )
    .env({"HF_HOME": "/models/huggingface"})
)


@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=3600,  # 1 hour for large model download
    cpu=4,  # More CPU for faster download
)
def download_madlad400():
    """Download MADLAD-400-3B model to persistent volume.

    This function downloads the model from HuggingFace Hub and saves it
    to the Modal persistent volume. The model is approximately 12GB.

    Returns:
        dict: Download status and timing information
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    MODEL_ID = "google/madlad400-3b-mt"
    MODEL_DIR = "/models/madlad-400-3b-mt"

    print(f"Starting download of {MODEL_ID}")
    print(f"Target directory: {MODEL_DIR}")

    start_time = time.perf_counter()

    # Check if model is already cached
    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"Model already exists at {MODEL_DIR}")
        files = os.listdir(MODEL_DIR)
        print(f"Files: {files}")

        # Verify model can be loaded
        try:
            print("Verifying model integrity...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
            print("Model verification successful!")

            return {
                "status": "already_cached",
                "model_id": MODEL_ID,
                "model_dir": MODEL_DIR,
                "files": files,
                "duration_s": round(time.perf_counter() - start_time, 2),
            }
        except Exception as e:
            print(f"Model verification failed: {e}")
            print("Re-downloading model...")

    # Download from HuggingFace
    print(f"Downloading model from HuggingFace: {MODEL_ID}")

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Download model
    print("Downloading model weights (this may take 10-30 minutes)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    # Save to persistent volume
    print(f"Saving model to {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

    # Commit volume changes
    volume.commit()

    download_time = time.perf_counter() - start_time
    files = os.listdir(MODEL_DIR)

    # Calculate total size
    total_size = 0
    for f in files:
        path = os.path.join(MODEL_DIR, f)
        if os.path.isfile(path):
            total_size += os.path.getsize(path)

    print(f"\nDownload completed!")
    print(f"Duration: {download_time:.2f}s")
    print(f"Total size: {total_size / 1e9:.2f} GB")
    print(f"Files: {files}")

    return {
        "status": "downloaded",
        "model_id": MODEL_ID,
        "model_dir": MODEL_DIR,
        "files": files,
        "size_gb": round(total_size / 1e9, 2),
        "duration_s": round(download_time, 2),
    }


@app.function(image=image, volumes={"/models": volume})
def verify_model():
    """Verify model files exist and can be loaded.

    Returns:
        dict: Verification status and file list
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    MODEL_DIR = "/models/madlad-400-3b-mt"

    print(f"Verifying model at {MODEL_DIR}")

    if not os.path.exists(MODEL_DIR):
        return {
            "status": "not_found",
            "model_dir": MODEL_DIR,
            "error": "Model directory does not exist",
        }

    files = os.listdir(MODEL_DIR)
    if not files:
        return {
            "status": "empty",
            "model_dir": MODEL_DIR,
            "error": "Model directory is empty",
        }

    # Check required files
    required_files = ["config.json", "tokenizer.json"]
    missing = [f for f in required_files if f not in files]

    if missing:
        return {
            "status": "incomplete",
            "model_dir": MODEL_DIR,
            "files": files,
            "missing": missing,
        }

    # Try to load model
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        print("Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

        # Test tokenization
        test_input = "<2zh> Hello"
        tokens = tokenizer(test_input, return_tensors="pt")

        print(f"Model loaded successfully!")
        print(f"Vocab size: {tokenizer.vocab_size}")

        return {
            "status": "verified",
            "model_dir": MODEL_DIR,
            "files": files,
            "vocab_size": tokenizer.vocab_size,
        }

    except Exception as e:
        return {
            "status": "error",
            "model_dir": MODEL_DIR,
            "files": files,
            "error": str(e),
        }


@app.function(image=image, volumes={"/models": volume})
def clear_cache():
    """Clear the model cache (use with caution).

    This function removes all cached model files from the persistent volume.

    Returns:
        dict: Cleanup status
    """
    import shutil

    MODEL_DIR = "/models/madlad-400-3b-mt"

    if not os.path.exists(MODEL_DIR):
        return {"status": "not_found", "model_dir": MODEL_DIR}

    print(f"Clearing model cache at {MODEL_DIR}")
    shutil.rmtree(MODEL_DIR)
    volume.commit()

    return {"status": "cleared", "model_dir": MODEL_DIR}


@app.local_entrypoint()
def main(action: str = "download"):
    """Run model management actions.

    Args:
        action: One of "download", "verify", or "clear"
    """
    if action == "download":
        result = download_madlad400.remote()
    elif action == "verify":
        result = verify_model.remote()
    elif action == "clear":
        result = clear_cache.remote()
    else:
        print(f"Unknown action: {action}")
        print("Available actions: download, verify, clear")
        return

    print(f"\nResult: {result}")
