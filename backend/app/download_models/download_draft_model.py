"""
Downloads the draft summarisation model (GGUF) from HuggingFace Hub
into the path declared in Config.DRAFT_MODEL_PATH.

Usage (from the app/ directory):
    python download_models/download_draft_model.py

The script is idempotent — if the file already exists at the target path
it prints a message and exits without re-downloading.
"""

import os
import sys
from pathlib import Path

# Allow imports from the app/ root when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402


def download_draft_model() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface-hub"
        )
        sys.exit(1)

    dest_dir = Path(Config.DRAFT_MODEL_PATH)
    dest_file = dest_dir / Config.DRAFT_MODEL_FILE

    if dest_file.exists():
        print(f"Model already present at:\n  {dest_file}")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading  : {Config.DRAFT_MODEL_NAME}")
    print(f"File         : {Config.DRAFT_MODEL_FILE}")
    print(f"Destination  : {dest_dir}")
    print()

    downloaded_path = hf_hub_download(
        repo_id=Config.DRAFT_MODEL_NAME,
        filename=Config.DRAFT_MODEL_FILE,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )

    print(f"\nSaved to: {downloaded_path}")


if __name__ == "__main__":
    download_draft_model()
