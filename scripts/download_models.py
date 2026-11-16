"""Explicit, opt-in model downloader. Downloads ONLY what is missing (already-present models are skipped).

  python scripts/download_models.py --list                 show every asset and whether it is present
  python scripts/download_models.py --missing              download what is missing from the recommended set (core llm tts-piper vision)
  python scripts/download_models.py --group grammar ser    download specific groups (still skips present ones)
  python scripts/download_models.py --missing --dry-run    show what WOULD be downloaded
  add --force to re-download something that is already present
Groups: core (SBERT, Whisper base.en), llm, tts-piper, vision, grammar, whisper-big, ser (3 x ~1.2 GB, heavy).
Requires internet and: pip install huggingface_hub.  Manual items (llama-server.exe, piper.exe, resemotenet.onnx, yolov8n.onnx) are only reported.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.paths import bootstrap  # noqa: E402

bootstrap()
import config as C  # noqa: E402
from shared import model_registry as M  # noqa: E402

RECOMMENDED = ["core", "llm", "tts-piper", "vision"]


def _hf(a):
    from huggingface_hub import snapshot_download
    snapshot_download(a.source, cache_dir=str(Path(a.location).parent), ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tflite", "*.mlmodel"])


def _hf_dir(a):
    from huggingface_hub import snapshot_download
    Path(a.location).mkdir(parents=True, exist_ok=True)
    snapshot_download(a.source, local_dir=str(a.location), ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tflite", "*.mlmodel"])


def _hf_file(a):
    from huggingface_hub import hf_hub_download
    dest = Path(a.location)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for f in a.files or []:
        hf_hub_download(a.source, f, local_dir=str(dest.parent))


def _url(a):
    dest = Path(a.location)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(a.source, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def _pip(a):
    print(f"    install with: {a.source}")


HANDLERS = {"hf": _hf, "hf_dir": _hf_dir, "hf_file": _hf_file, "url": _url, "pip": _pip}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--missing", action="store_true", help="download what is missing from the recommended groups")
    ap.add_argument("--group", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if a.list or not (a.missing or a.group or a.all):
        for x in M.ASSETS:
            print(f"[{'present' if M.is_present(x) else 'MISSING':7}] [{x.group:10}] {x.key:24} {x.size:>9}  -> {x.location}")
        print("\nRun with --missing to fetch the recommended missing models, or --group NAME ...")
        return 0

    groups = list(a.group) + (RECOMMENDED if a.missing else [])
    assets = M.ASSETS if a.all else M.by_group(groups)
    C.apply_runtime_env(offline=False)    # downloading is the one place network is allowed
    fails = 0
    for x in assets:
        if M.is_present(x) and not a.force:
            print(f"[skip] {x.key}: already present")
            continue
        if x.kind == "manual":
            print(f"[MANUAL] {x.key}: {x.note or x.purpose}\n         source: {x.source}\n         place at: {x.location}")
            continue
        print(f"[download] {x.key} [{x.kind}] {x.source}\n           -> {x.location}")
        if a.dry_run:
            continue
        try:
            HANDLERS[x.kind](x)
            print("           ok" if M.is_present(x) else "           finished, but the folder/file does not look complete - check it")
        except Exception as e:
            fails += 1
            print(f"           FAILED: {type(e).__name__}: {e}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
