import argparse
import pathlib

import pytest

from lexbrowser_eval.online_mind2web.cli import (
    build_rollout_command,
    resolve_policy_metadata,
    write_config,
)


def test_resolve_policy_metadata_records_checkpoint(tmp_path):
    artifact = tmp_path / "checkpoint"
    artifact.mkdir()
    digest = "a" * 64
    (artifact / "model.safetensors.sha256").write_text(digest + "  model.safetensors\n")

    result = resolve_policy_metadata(
        argparse.Namespace(
            policy_label="Qwen3-8B WebVoyager GRPO step 150",
            policy_artifact=artifact,
            policy_sha256=digest,
        )
    )

    assert result == {
        "label": "Qwen3-8B WebVoyager GRPO step 150",
        "artifact_dir": str(artifact.resolve()),
        "safetensors_sha256": digest,
    }


def test_resolve_policy_metadata_rejects_sidecar_mismatch(tmp_path):
    artifact = tmp_path / "checkpoint"
    artifact.mkdir()
    (artifact / "model.safetensors.sha256").write_text("a" * 64)

    with pytest.raises(ValueError, match="does not match"):
        resolve_policy_metadata(
            argparse.Namespace(
                policy_label="checkpoint",
                policy_artifact=artifact,
                policy_sha256="b" * 64,
            )
        )


def test_build_rollout_command_records_requested_concurrency():
    command = build_rollout_command(
        pathlib.Path("/benchmark"),
        pathlib.Path("/benchmark/config.yaml"),
        "local",
        "20260718_230000",
        rollout_concurrency=1,
    )

    assert command[command.index("--concurrency") + 1] == "1"


def test_write_config_records_explicit_policy_temperature(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    config = tmp_path / "config.yaml"

    write_config(config, checkout, policy_temperature=0.0)

    assert "temperature: 0.0" in config.read_text(encoding="utf-8")
    assert config.read_text(encoding="utf-8") == (checkout / "config.yaml").read_text(
        encoding="utf-8"
    )
