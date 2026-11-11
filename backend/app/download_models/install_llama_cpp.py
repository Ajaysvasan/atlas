"""Installs llama-cpp-python with the correct hardware backend.

    uv run --no-sync python download_models/install_llama_cpp.py            # detect
    uv run --no-sync python download_models/install_llama_cpp.py --backend cuda
    uv run --no-sync python download_models/install_llama_cpp.py --dry-run  # print the command only

See README.md in this directory.
"""

import argparse
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys

PACKAGE = "llama-cpp-python"
BACKENDS = ("auto", "cuda", "rocm", "metal", "cpu")
# Blackwell (compute capability 12.x, e.g. RTX 50xx) needs CUDA 12.8 or later to compile for.
BLACKWELL_MIN_CUDA = (12, 8)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True)
    except OSError as error:
        return subprocess.CompletedProcess(cmd, 127, "", str(error))


def has_nvidia_gpu() -> bool:
    """True when nvidia-smi is present and returns exit code 0 (the driver is loaded)."""
    if shutil.which("nvidia-smi") is None:
        return False
    return _run(["nvidia-smi"]).returncode == 0


def nvidia_hardware_present() -> bool:
    """True when the PCI bus shows an NVIDIA display device, whether or not its driver is loaded."""
    if shutil.which("lspci") is None:
        return False
    result = _run(["lspci"])
    return any(
        "nvidia" in line.lower() and ("vga" in line.lower() or "3d" in line.lower())
        for line in result.stdout.splitlines()
    )


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
    """The CUDA toolkit version from nvcc (e.g. '12.8'); None without nvcc."""
    if shutil.which("nvcc") is None:
        return None
    for line in _run(["nvcc", "--version"]).stdout.splitlines():
        if "release" in line.lower():
            # "Cuda compilation tools, release 12.8, V12.8.93"
            return line.split("release")[1].strip().split(",")[0].strip()
    return None


def detect_compute_capability() -> str | None:
    """The first GPU's compute capability as CMake wants it ('8.6' -> '86'); None if unknown."""
    result = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if result.returncode != 0:
        return None
    lines = result.stdout.split()
    if not lines:
        return None
    capability = lines[0].strip()
    return capability.replace(".", "") if capability.replace(".", "").isdigit() else None


def installed_version() -> str | None:
    """The llama-cpp-python version already installed, so a rebuild keeps it."""
    try:
        return importlib.metadata.version(PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_tuple(version: str) -> tuple:
    try:
        return tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        return ()


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

def cmake_args(backend: str, architecture: str | None = None) -> str:
    """The CMAKE_ARGS that select a backend; empty for the CPU."""
    if backend == "cuda":
        args = "-DGGML_CUDA=on"
        if architecture:
            args += f" -DCMAKE_CUDA_ARCHITECTURES={architecture}"
        return args
    return {"rocm": "-DGGML_HIPBLAS=on", "metal": "-DGGML_METAL=on"}.get(backend, "")


def install_command(requirement: str, has_pip: bool, uv: str | None, rebuild: bool) -> list[str]:
    """pip when this interpreter has it, else uv pip aimed at this interpreter (bug 9.9: uv venvs have no pip).

    A rebuild replaces llama-cpp-python alone, so numpy and the rest of the
    environment stay exactly as uv.lock left them; a fresh install pulls its
    dependencies. The cache is skipped either way, or a cached CPU wheel would
    be reused instead of compiling for the chosen backend.
    """
    if has_pip:
        options = ["--no-cache-dir", *(["--force-reinstall", "--no-deps"] if rebuild else [])]
        return [sys.executable, "-m", "pip", "install", *options, requirement]
    if uv:
        options = ["--no-cache", *(["--reinstall-package", PACKAGE] if rebuild else [])]
        return [uv, "pip", "install", "--python", sys.executable, *options, requirement]
    raise SystemExit(
        "Neither pip nor uv is available to install with. Install uv (https://docs.astral.sh/uv/) "
        "or run this with an interpreter that has pip."
    )


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if has_nvidia_gpu():
        return "cuda"
    if nvidia_hardware_present():
        raise SystemExit(
            "An NVIDIA GPU is installed but its driver is not loaded (nvidia-smi fails).\n"
            "On Linux with Secure Boot this is usually an unsigned kernel module: enroll the akmods key "
            "(sudo mokutil --import /etc/pki/akmods/certs/public_key.der, reboot, Enroll MOK) or turn "
            "Secure Boot off, then run this again.\n"
            "To install for the CPU anyway: --backend cpu"
        )
    if has_amd_gpu():
        return "rocm"
    if has_apple_silicon():
        return "metal"
    return "cpu"


def check_cuda_toolkit(architecture: str | None) -> None:
    """Refuse a CUDA build that cannot succeed, before spending minutes on it."""
    version = detect_cuda_version()
    if version is None:
        raise SystemExit(
            "A CUDA build needs the CUDA toolkit (nvcc), which is not on PATH. Install it from "
            "https://developer.nvidia.com/cuda-downloads and make sure `nvcc --version` works."
        )
    if architecture and architecture.startswith("12") and _version_tuple(version) < BLACKWELL_MIN_CUDA:
        raise SystemExit(
            f"This GPU (compute capability {architecture[:-1]}.{architecture[-1]}) needs CUDA "
            f"{'.'.join(map(str, BLACKWELL_MIN_CUDA))} or later to build for; nvcc is {version}."
        )


def install(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Install llama-cpp-python for this machine's hardware")
    parser.add_argument("--backend", choices=BACKENDS, default="auto")
    parser.add_argument("--version", help="llama-cpp-python version (default: the installed one, else latest)")
    parser.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
    args = parser.parse_args(argv)

    print("=== llama-cpp-python installer ===\n")
    backend = choose_backend(args.backend)
    architecture = detect_compute_capability() if backend == "cuda" else None
    if backend == "cuda" and not args.dry_run:
        check_cuda_toolkit(architecture)
    installed = installed_version()
    version = args.version or installed
    requirement = f"{PACKAGE}=={version}" if version else PACKAGE
    env = {"CMAKE_ARGS": cmake_args(backend, architecture)} if backend != "cpu" else {}
    command = install_command(requirement, importlib.util.find_spec("pip") is not None, shutil.which("uv"),
                              rebuild=installed is not None)

    print(f"Backend  : {backend}" + (f" (GPU architecture {architecture})" if architecture else ""))
    print(f"Running  : {' '.join(f'{k}={v!r}' for k, v in env.items())} {' '.join(command)}\n")
    if args.dry_run:
        return 0
    result = subprocess.run(command, env={**os.environ, **env})
    if result.returncode != 0:
        print("\nInstallation failed; see the output above.")
        return result.returncode
    print("\nDone. Verify with:")
    print("  uv run --no-sync python -c \"import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())\"")
    return 0


if __name__ == "__main__":
    sys.exit(install())
