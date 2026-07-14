#!/usr/bin/env bash
# One Slurm task per node. Emits a secret-free hardware and shared-storage fact.
set -Eeuo pipefail

: "${LEXBROWSER_RUN_DIR:?LEXBROWSER_RUN_DIR is required}"
: "${LEXBROWSER_EXPECTED_GPUS:?LEXBROWSER_EXPECTED_GPUS is required}"
: "${LEXBROWSER_EXPECTED_GPU_FAMILY:?LEXBROWSER_EXPECTED_GPU_FAMILY is required}"
: "${LEXBROWSER_MIN_FREE_MIB:?LEXBROWSER_MIN_FREE_MIB is required}"

host="$(hostname -s)"
output_dir="$LEXBROWSER_RUN_DIR/preflight/nodes"
mkdir -p "$output_dir"

python3 - "$output_dir/$host.json" <<'PY'
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

output = Path(sys.argv[1])
expected = int(os.environ["LEXBROWSER_EXPECTED_GPUS"])
expected_family = os.environ["LEXBROWSER_EXPECTED_GPU_FAMILY"].casefold()
minimum_free = int(os.environ["LEXBROWSER_MIN_FREE_MIB"])

def command(*args: str) -> tuple[int, str]:
    try:
        completed = subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        return 127, "missing"
    return completed.returncode, completed.stdout.strip()

status, raw_gpus = command(
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.free,uuid,driver_version",
    "--format=csv,noheader,nounits",
)
gpus = []
if status == 0:
    for line in raw_gpus.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            name, total, free, uuid, driver = fields
            gpus.append({"name": name, "memory_total_mib": int(total), "memory_free_mib": int(free), "uuid": uuid, "driver": driver})

storage = shutil.disk_usage(os.environ["LEXBROWSER_RUN_DIR"])
ib_status, ib_output = command("ibdev2netdev")
payload = {
    "schema_version": 1,
    "collected_at_unix": time.time(),
    "hostname": os.uname().nodename,
    "expected_gpus": expected,
    "expected_gpu_family": expected_family,
    "minimum_free_mib": minimum_free,
    "gpus": gpus,
    "nvidia_smi_status": status,
    "nvidia_smi_error": raw_gpus if status else "",
    "shared_storage": {"path": os.environ["LEXBROWSER_RUN_DIR"], "total_bytes": storage.total, "free_bytes": storage.free},
    "slurm": {key: os.environ.get(key, "") for key in ("SLURM_JOB_ID", "SLURM_PROCID", "SLURMD_NODENAME")},
    "infiniband": {"available": ib_status == 0, "mapping": ib_output if ib_status == 0 else ""},
}
payload["checks"] = {
    "gpu_count": len(gpus) == expected,
    "gpu_family": bool(gpus) and all(
        expected_family in {"any", "nvidia"} or expected_family in gpu["name"].casefold()
        for gpu in gpus
    ),
    "gpu_memory": bool(gpus) and all(gpu["memory_free_mib"] >= minimum_free for gpu in gpus),
    "shared_storage_writable": True,
}
temporary = output.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps({"hostname": payload["hostname"], "checks": payload["checks"]}, sort_keys=True))
PY
