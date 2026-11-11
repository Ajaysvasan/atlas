"""install_llama_cpp.py builds the right backend with whatever installer the venv has (bug 9.9), and refuses doomed builds."""

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "download_models" / "install_llama_cpp.py"
_spec = importlib.util.spec_from_file_location("install_llama_cpp", _PATH)
installer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(installer)


class TestCommand:
    def test_a_uv_venv_without_pip_uses_uv_aimed_at_this_interpreter(self):
        command = installer.install_command("llama-cpp-python==0.3.36", has_pip=False, uv="/bin/uv", rebuild=True)
        assert command[:5] == ["/bin/uv", "pip", "install", "--python", sys.executable]
        assert "--reinstall-package" in command and "--no-cache" in command
        assert command[-1] == "llama-cpp-python==0.3.36"

    def test_a_rebuild_replaces_only_llama_cpp_python(self):
        uv = installer.install_command("x", has_pip=False, uv="/bin/uv", rebuild=True)
        pip = installer.install_command("x", has_pip=True, uv=None, rebuild=True)
        assert uv[uv.index("--reinstall-package") + 1] == "llama-cpp-python"
        assert "--no-deps" in pip and "--force-reinstall" in pip
        assert "--force-reinstall" not in uv  # that would reinstall numpy and friends too

    def test_a_fresh_install_pulls_its_dependencies(self):
        assert "--reinstall-package" not in installer.install_command("x", False, "/bin/uv", rebuild=False)
        assert "--no-deps" not in installer.install_command("x", True, None, rebuild=False)

    def test_pip_is_used_when_the_interpreter_has_it(self):
        assert installer.install_command("x", True, "/bin/uv", rebuild=False)[:4] == [
            sys.executable, "-m", "pip", "install"]

    def test_with_neither_installer_it_says_so(self):
        with pytest.raises(SystemExit, match="Neither pip nor uv"):
            installer.install_command("x", False, None, rebuild=False)

    @pytest.mark.parametrize("backend, architecture, expected", [
        ("cuda", "120", "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120"),
        ("cuda", None, "-DGGML_CUDA=on"),
        ("rocm", None, "-DGGML_HIPBLAS=on"),
        ("metal", None, "-DGGML_METAL=on"),
        ("cpu", None, ""),
    ])
    def test_cmake_args(self, backend, architecture, expected):
        assert installer.cmake_args(backend, architecture) == expected


class TestDetection:
    def test_a_gpu_whose_driver_is_not_loaded_stops_with_the_fix(self, monkeypatch):
        monkeypatch.setattr(installer, "has_nvidia_gpu", lambda: False)
        monkeypatch.setattr(installer, "nvidia_hardware_present", lambda: True)
        with pytest.raises(SystemExit, match="driver is not loaded") as stopped:
            installer.choose_backend("auto")
        assert "mokutil --import" in str(stopped.value) and "--backend cpu" in str(stopped.value)

    def test_a_working_nvidia_driver_means_cuda(self, monkeypatch):
        monkeypatch.setattr(installer, "has_nvidia_gpu", lambda: True)
        assert installer.choose_backend("auto") == "cuda"

    def test_an_explicit_backend_is_not_second_guessed(self, monkeypatch):
        monkeypatch.setattr(installer, "has_nvidia_gpu", lambda: (_ for _ in ()).throw(AssertionError("probed")))
        assert installer.choose_backend("cpu") == "cpu"

    def test_the_compute_capability_becomes_a_cmake_architecture(self, monkeypatch):
        monkeypatch.setattr(installer, "_run", lambda cmd: installer.subprocess.CompletedProcess(cmd, 0, "12.0\n", ""))
        assert installer.detect_compute_capability() == "120"

    def test_no_driver_means_no_architecture(self, monkeypatch):
        monkeypatch.setattr(installer, "_run", lambda cmd: installer.subprocess.CompletedProcess(cmd, 9, "", "fail"))
        assert installer.detect_compute_capability() is None

    def test_a_missing_executable_is_a_failed_command_not_a_crash(self):
        assert installer._run(["/nonexistent/nvidia-smi"]).returncode == 127


class TestToolkitCheck:
    def test_no_nvcc_is_refused_before_building(self, monkeypatch):
        monkeypatch.setattr(installer, "detect_cuda_version", lambda: None)
        with pytest.raises(SystemExit, match="needs the CUDA toolkit"):
            installer.check_cuda_toolkit("120")

    def test_blackwell_needs_cuda_12_8(self, monkeypatch):
        monkeypatch.setattr(installer, "detect_cuda_version", lambda: "12.4")
        with pytest.raises(SystemExit, match=r"compute capability 12\.0\) needs CUDA 12\.8"):
            installer.check_cuda_toolkit("120")

    @pytest.mark.parametrize("version, architecture", [("12.8", "120"), ("13.0", "120"), ("12.4", "89"), ("12.4", None)])
    def test_a_capable_toolkit_passes(self, monkeypatch, version, architecture):
        monkeypatch.setattr(installer, "detect_cuda_version", lambda: version)
        installer.check_cuda_toolkit(architecture)


def test_a_dry_run_prints_the_command_and_runs_nothing(monkeypatch, capsys):
    monkeypatch.setattr(installer, "choose_backend", lambda requested: "cuda")
    monkeypatch.setattr(installer, "detect_compute_capability", lambda: "120")
    monkeypatch.setattr(installer, "installed_version", lambda: "0.3.36")
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    assert installer.install(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "CMAKE_ARGS='-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120'" in out
    assert "llama-cpp-python==0.3.36" in out
