"""
Installs llama-cpp-python with the correct hardware backend:
  - NVIDIA GPU (CUDA)  → builds with GGML_CUDA=on
  - AMD GPU   (ROCm)   → builds with GGML_HIPBLAS=on
  - Apple Silicon      → builds with GGML_METAL=on
  - CPU fallback       → plain pip install (no GPU acceleration)

Usage (from any directory):
    python download_models/install_llama_cpp.py

The script detects the hardware automatically — no flags needed.
"""

import os
import platform
import shutil
import subprocess
import sys


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def has_nvidia_gpu() -> bool:
    """True when nvidia-smi is present and returns exit code 0."""
    if shutil.which("nvidia-smi") is None:
        return False
    result = _run(["nvidia-smi"])
    return result.returncode == 0


def has_amd_gpu() -> bool:
    """True when rocm-smi is present OR the ROCm device node exists."""
    if shutil.which("rocm-smi") is not None:
        result = _run(["rocm-smi"])
        if result.returncode == 0:
            return True
    return os.path.exists("/dev/kfd")


def has_apple_silicon() -> bool:
    """True on macOS running on ARM (M1/M2/M3/M4)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def detect_cuda_version() -> str | None:
    """
    Returns the CUDA version string (e.g. '12.1') if nvcc or nvidia-smi
    can report it, otherwise None.
    """
    # Try nvcc first — most reliable
    if shutil.which("nvcc"):
        result = _run(["nvcc", "--version"])
        for line in result.stdout.splitlines():
            if "release" in line.lower():
                # "Cuda compilation tools, release 12.1, V12.1.105"
                parts = line.split("release")
                if len(parts) > 1:
                    return parts[1].strip().split(",")[0].strip()

    # Fall back to nvidia-smi
    result = _run(["nvidia-smi"])
    for line in result.stdout.splitlines():
        if "CUDA Version:" in line:
            return line.split("CUDA Version:")[-1].strip().split()[0]

    return None


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def pip_install(env: dict[str, str], package: str) -> None:
    merged_env = {**os.environ, **env}
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]
    print(f"Running: {' '.join(f'{k}={v}' for k, v in env.items())} {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=merged_env)
    if result.returncode != 0:
        print("\nInstallation failed — see output above.")
        sys.exit(result.returncode)


def install() -> None:
    print("=== llama-cpp-python installer ===\n")

    if has_nvidia_gpu():
        cuda_ver = detect_cuda_version()
        print(f"Detected : NVIDIA GPU  (CUDA {cuda_ver or 'unknown'})")
        print("Backend  : GGML_CUDA=on  (builds with CUDA support)\n")
        pip_install({"CMAKE_ARGS": "-DGGML_CUDA=on"}, "llama-cpp-python")

    elif has_amd_gpu():
        print("Detected : AMD GPU  (ROCm)")
        print("Backend  : GGML_HIPBLAS=on  (builds with HIP/ROCm support)\n")
        pip_install({"CMAKE_ARGS": "-DGGML_HIPBLAS=on"}, "llama-cpp-python")

    elif has_apple_silicon():
        print("Detected : Apple Silicon  (Metal)")
        print("Backend  : GGML_METAL=on  (builds with Metal GPU support)\n")
        pip_install({"CMAKE_ARGS": "-DGGML_METAL=on"}, "llama-cpp-python")

    else:
        print("Detected : CPU only  (no GPU found)")
        print("Backend  : plain CPU build\n")
        pip_install({}, "llama-cpp-python")

    print("\nDone. Verify with:")
    print("  python -c \"from llama_cpp import Llama; print('llama-cpp-python OK')\"")


if __name__ == "__main__":
    install()
