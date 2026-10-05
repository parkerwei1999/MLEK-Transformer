"""Check the running Python environment against pinned requirements files: every `name==version` pin
is installed at exactly that version; with --cuda, torch sees a GPU and the ExecuTorch Arm backend
imports. Exits non-zero on any mismatch.
Usage: python check_env.py [--cuda] <requirements.txt>...
"""
import argparse
import importlib.metadata as md
import re
import sys


def pins(path):
    out = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if line:
            m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==(\S+)", line)
            assert m, f"{path}: not a name==version pin: {line!r}"
            out.append(m.groups())
    return out


def main(args):
    bad = []
    for req in args.requirements:
        for name, want in pins(req):
            try:
                have = md.version(name)
            except md.PackageNotFoundError:
                have = None
            ok = have == want
            print(f"{'ok ' if ok else 'BAD'} {name}=={want}" + ("" if ok else f" (installed: {have})"))
            if not ok:
                bad.append(name)
    if args.cuda:
        import torch
        import executorch.backends.arm.ethosu  # noqa: F401
        import executorch.backends.arm.quantizer  # noqa: F401
        print(f"cuda available={torch.cuda.is_available()} version={torch.version.cuda}"
              + (f" device={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
        print("executorch arm backend import ok")
        if not torch.cuda.is_available():
            bad.append("cuda")
    if bad:
        sys.exit(f"environment check FAILED: {', '.join(bad)}")
    print("environment check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true", default=False, help="also require a CUDA device and the ExecuTorch Arm backend")
    ap.add_argument("requirements", nargs="+")
    main(ap.parse_args())
