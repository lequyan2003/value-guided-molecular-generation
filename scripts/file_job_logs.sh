#!/usr/bin/env bash
# Files flat scheduler job logs (logs/<name>-<id>.{out,err,res}) into logs/<machine>/. The shipped GPU tooling
# writes and fetches them flat in logs/ (GAIVI: sbatch --output=logs/%x-%j.out); run this after a fetch.
# Usage: scripts/file_job_logs.sh [gaivi|5090|4090]   (default gaivi)
set -euo pipefail
machine="${1:-gaivi}"
case "$machine" in gaivi|5090|4090) ;; *) echo "machine must be gaivi, 5090 or 4090" >&2; exit 2;; esac
mkdir -p "logs/$machine"
find logs -maxdepth 1 -type f ! -name '.gitkeep' -exec mv -v {} "logs/$machine/" \;
