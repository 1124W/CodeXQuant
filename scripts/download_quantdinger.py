"""Download the pinned QuantDinger backend using a bounded sparse Git checkout."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/OpenByteInc/QuantDinger.git"
COMMIT = "d1cf2681bc84fba2551973b4bcab011ad483e8ad"


def run(args, *, cwd=None):
    print("Running: " + " ".join(str(part) for part in args), flush=True)
    subprocess.run(args, cwd=cwd, check=True, timeout=180)


def download(destination):
    destination = destination.resolve()
    if destination.exists():
        current = subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"],
                                 check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        if current != COMMIT:
            raise ValueError(f"Existing checkout is {current}; expected {COMMIT}. Choose a new --destination.")
        registry = destination / "backend_api_python/app/services/factors/registry.py"
        if not registry.is_file():
            run(["git", "-C", str(destination), "sparse-checkout", "set", "backend_api_python", "docs/trading"])
        if not registry.is_file():
            raise FileNotFoundError("Pinned checkout exists but the QuantDinger factor source is incomplete")
        print(f"Pinned QuantDinger checkout already present: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "-c", "http.version=HTTP/1.1", "-c", "http.lowSpeedLimit=1024", "-c",
         "http.lowSpeedTime=30", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         REPOSITORY, str(destination)])
    current = subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if current != COMMIT:
        run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", COMMIT])
        run(["git", "-C", str(destination), "checkout", "--detach", COMMIT])
    run(["git", "-C", str(destination), "sparse-checkout", "set", "backend_api_python", "docs/trading"])
    print(f"Downloaded QuantDinger backend and trading documentation at {COMMIT}", flush=True)
    print("Frontend, PostgreSQL, Redis, workers and live trading are not started by this command.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "external" / "QuantDinger-repo")
    args = parser.parse_args()
    try:
        download(args.destination)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
