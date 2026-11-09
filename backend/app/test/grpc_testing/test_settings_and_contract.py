"""Server settings refuse unsafe setups; filenames are made safe; the bindings match the shared .proto."""

import importlib.util
from pathlib import Path

import pytest

from grpc_server.settings import GrpcSettings, SettingsError
from grpc_server.uploads import extension, safe_filename

TOKEN = "x" * 32


class TestSettings:
    @pytest.mark.parametrize("token", ["", "short"])
    def test_a_missing_or_weak_service_token_refuses_to_start(self, token):
        with pytest.raises(SettingsError, match="ATLAS_GRPC_SERVICE_TOKEN"):
            GrpcSettings(service_token=token)

    def test_a_public_bind_needs_tls(self):
        with pytest.raises(SettingsError, match="TLS"):
            GrpcSettings(service_token=TOKEN, bind="0.0.0.0")
        GrpcSettings(service_token=TOKEN, bind="0.0.0.0", tls_cert="c.pem", tls_key="k.pem")

    def test_mtls_needs_a_server_certificate(self):
        with pytest.raises(SettingsError):
            GrpcSettings(service_token=TOKEN, tls_client_ca="ca.pem")

    def test_the_environment_supplies_everything(self, monkeypatch):
        monkeypatch.setenv("ATLAS_GRPC_SERVICE_TOKEN", TOKEN)
        monkeypatch.setenv("ATLAS_GRPC_PORT", "50999")
        monkeypatch.setenv("ATLAS_UPLOAD_MAX_FILES", "3")
        settings = GrpcSettings.from_environment()
        assert (settings.port, settings.max_files, settings.bind) == (50999, 3, "127.0.0.1")
        monkeypatch.setenv("ATLAS_GRPC_PORT", "not-a-port")
        with pytest.raises(SettingsError):
            GrpcSettings.from_environment()


class TestFilenames:
    @pytest.mark.parametrize("raw, safe", [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system.ini", "system.ini"),
        ("/abs/path/notes.md", "notes.md"),
        ("bad\x00name\n.txt", "bad_name_.txt"),
        ("   ", "upload"),
        ("..", "upload"),
        ("", "upload"),
    ])
    def test_names_lose_every_directory_and_control_character(self, raw, safe):
        assert safe_filename(raw) == safe

    def test_long_names_keep_their_extension(self):
        name = safe_filename("a" * 500 + ".docx")
        assert len(name) <= 200 and name.endswith(".docx")

    def test_extensions_are_lower_case(self):
        assert extension("Paper.PDF") == "pdf" and extension("noext") == ""


def test_the_checked_in_bindings_match_the_shared_proto():
    script = Path(__file__).resolve().parents[2] / "scripts" / "generate_grpc.py"
    spec = importlib.util.spec_from_file_location("generate_grpc", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.stale() == [], "run: uv run python scripts/generate_grpc.py"
