import argparse

import pytest

from lexbrowser_eval.online_mind2web.cli import resolve_policy_metadata


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
