"""Verify the bundled ICP inputs; this artifact requires no data downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ipj.reproduction import verify_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify bundled SHA-256 checksums (also the default).")
    parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    failures = verify_inputs(root, manifest["files"])
    print(json.dumps({"status": "failed" if failures else "passed", "inputs": len(manifest["files"]),
                      "downloads_required": False, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
