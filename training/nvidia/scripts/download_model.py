#!/usr/bin/env python3
"""Fetch a revision-pinned Hugging Face model and record only non-secret facts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    if args.manifest.exists():
        existing = json.loads(args.manifest.read_text(encoding="utf-8"))
        if (
            existing.get("model_id") == args.model_id
            and existing.get("requested_revision") == args.revision
        ):
            if not args.output.is_dir() or tree_hash(args.output) != existing.get(
                "snapshot_sha256"
            ):
                raise SystemExit("cached model directory no longer matches its manifest")
            print(json.dumps(existing, sort_keys=True))
            return
        raise SystemExit("model manifest disagrees with the requested model or revision")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(
            "model directory is non-empty without a completed manifest; "
            "remove it or use a new run id"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    resolved = Path(
        snapshot_download(repo_id=args.model_id, revision=args.revision, local_dir=args.output)
    )
    resolved_revision = args.revision
    git_ref = resolved / ".cache" / "huggingface" / "download" / "refs" / "main"
    if git_ref.is_file():
        resolved_revision = git_ref.read_text(encoding="utf-8").strip() or args.revision
    payload = {
        "schema_version": 1,
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "path": str(resolved),
        "snapshot_sha256": tree_hash(resolved),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
