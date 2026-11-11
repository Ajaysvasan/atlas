"""Where the models run: llama.cpp layers on the GPU when it can, torch models on CUDA when present, CPU otherwise."""

from typing import Callable, TypeVar

from config import Config, get_logger

logger = get_logger(__name__)

T = TypeVar("T")

CPU = "cpu"
CUDA = "cuda"


def llama_gpu_offload() -> bool:
    """True when the installed llama-cpp-python is a GPU build and its backend found a device."""
    try:
        import llama_cpp
    except ImportError:
        return False
    try:
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as error:
        logger.warning("Could not ask llama.cpp about GPU offload: %s", error)
        return False


def gpu_layers(configured: int | None = None, supported: Callable[[], bool] | None = None) -> int:
    """The n_gpu_layers to request: the configured value when offload is possible, else 0."""
    configured = Config.MODEL_GPU_LAYERS if configured is None else configured
    if configured == 0:
        return 0
    supported = llama_gpu_offload if supported is None else supported
    return configured if supported() else 0


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception as error:
        logger.warning("Could not ask torch about CUDA: %s", error)
        return False


def torch_device(configured: str | None = None, available: Callable[[], bool] | None = None) -> str:
    """The device for the embedder and reranker: CUDA for auto/cuda when it is there, else the CPU."""
    configured = Config.TORCH_DEVICE if configured is None else configured
    if configured == CPU:
        return CPU
    available = cuda_available if available is None else available
    if available():
        return CUDA
    if configured == CUDA:
        logger.warning("ATLAS_TORCH_DEVICE=cuda but CUDA is not available; using the CPU")
    return CPU


def gpu_name(available: Callable[[], bool] | None = None) -> str | None:
    """The first CUDA device's name, for status; None without one."""
    available = cuda_available if available is None else available
    if not available():
        return None
    try:
        import torch

        return str(torch.cuda.get_device_name(0))
    except Exception:
        return None


def release_cuda_cache() -> None:
    """Hand freed CUDA memory back to the driver, after a GPU model is dropped."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_on_device(factory: Callable[[str], T], what: str, device: str | None = None) -> tuple[T, str]:
    """factory(device), falling back to the CPU when the GPU load fails; returns the model and where it is."""
    device = torch_device() if device is None else device
    if device == CPU:
        return factory(CPU), CPU
    try:
        return factory(device), device
    except Exception as error:
        logger.warning("Could not load the %s on %s (%s); using the CPU", what, device, error)
        return factory(CPU), CPU
