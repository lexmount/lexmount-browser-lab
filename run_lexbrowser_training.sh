#!/usr/bin/env bash
set -Eeuo pipefail

mode="${1:-train}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/training/scripts/run_lexbrowser_grpo.sh" "$mode"
