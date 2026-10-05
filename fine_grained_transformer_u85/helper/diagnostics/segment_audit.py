"""Audit the piece (segment) latency of a lowered configuration against the whole graph.

The latency flow (compile_and_profile.sh, default mode) lowers a model in pieces (embedding / one block / output; Swin also
per block group and patch merging) and reports sum(piece NPU-active cycles x repetitions). This script checks two things
for each given tag that already has piece results:

1. Piece inputs. A piece receives every tensor that crosses its boundary as a graph input, quantized on the host. When
   a piece takes a boundary tensor at a width its producer piece does not output (e.g. the block output is INT8 but the
   next LayerNorm reads it at INT16 under LN-16 / LN-32 / PCS), the conversion, which the whole graph runs on the NPU, is
   done on the host and never timed. Counted per piece instance against its producer's outputs (the previous piece for
   the first instance, its own outputs for the repeats), and per forward pass. The first piece's inputs are model inputs.
2. Whole graph. With --run, the same lowering arguments are compiled as one graph (compile_and_profile.sh --whole, tag
   <tag>_whole) and profiled on the FVP; the table compares its NPU-active ms with the piece total per MAC count. The
   whole graph must fit the FVP memory window (DeiT, Swin-T, small Whisper sizes); larger models fail at invoke.

usage: segment_audit.py <tag>... [--macs 256 512 1024 2048] [--run]
  without --run: piece-input check, plus the comparison for whole-graph cells that already exist
  with --run:    also lower and profile the missing whole graphs (needs the CUDA + FVP setup of compile_and_profile.sh)
Environment: as compile_and_profile.sh (U85_WORK from fvp/env.sh; default <fine_grained_transformer_u85>/out/fvp).
"""
import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # fine_grained_transformer_u85/
WORK = Path(os.environ.get("U85_WORK", ROOT / "out" / "fvp"))
CLOCK_HZ = {256: 500e6, 512: 1e9, 1024: 1e9, 2048: 1e9}  # fvp/env.sh U85_CLOCK_HZ


def parse_lower_args(text: str) -> tuple[str, list[str]]:
    """lower.args is '<mode> <args...>' written with "$*" (quoting lost); option values may contain spaces
    (--recipe 'Default A8W8'), so split on ' --<flag>' boundaries instead of whitespace."""
    mode, _, rest = text.strip().partition(" ")
    argv = []
    for chunk in re.split(r" (?=--[a-z][a-z0-9-]*(?: |$))", " " + rest):
        chunk = chunk.strip()
        if chunk:
            flag, _, val = chunk.partition(" ")
            argv += [flag, val] if val else [flag]
    return mode, argv


def graph_io(tosa_path: Path) -> tuple[list[tuple[str, tuple[int, ...]]], list[tuple[str, tuple[int, ...]]]]:
    """(inputs, outputs) of a TOSA graph as (dtype, shape) pairs."""
    from tosa.TosaGraph import TosaGraph
    from tosa import DType
    names = {v: k for k, v in vars(DType.DType).items() if not k.startswith("_")}
    blk = TosaGraph.GetRootAs(tosa_path.read_bytes(), 0).Regions(0).Blocks(0)
    tensors = {blk.Tensors(i).Name(): blk.Tensors(i) for i in range(blk.TensorsLength())}
    def sig(name):
        t = tensors[name]
        return names[t.Type()], tuple(int(d) for d in t.ShapeAsNumpy())
    return ([sig(blk.Inputs(i)) for i in range(blk.InputsLength())],
            [sig(blk.Outputs(i)) for i in range(blk.OutputsLength())])


def untimed(inputs, producer_outputs) -> int:
    """Input widths of a boundary tensor (matched by shape) that the producer piece does not output."""
    produced = defaultdict(set)
    for dt, shape in producer_outputs:
        produced[shape].add(dt)
    return sum(1 for dt, shape in inputs if dt not in produced[shape])


def piece_check(tag: str) -> tuple[list[str], int]:
    """Rows describing each piece's inputs, and the number of untimed host-side conversions per forward pass."""
    d = WORK / "lower" / tag
    rows, per_pass, prev_out = [], 0, None
    for line in (d / "pieces.tsv").read_text().splitlines():
        piece, reps = line.split("\t")[:2]
        reps = int(reps)
        parts = sorted((d / piece / "tosa").glob("*.tosa"))
        ins, outs = graph_io(parts[0]) if parts else ([], [])
        if len(parts) > 1:
            outs = graph_io(parts[-1])[1]  # host ops between partitions (e.g. Swin shifts): the piece output is the last one's
        first = 0 if prev_out is None else untimed(ins, prev_out)
        again = untimed(ins, outs) if reps > 1 else 0
        n = first + again * (reps - 1)
        per_pass += n
        by_shape = defaultdict(list)
        for dt, shape in ins:
            by_shape[shape].append(dt)
        desc = ", ".join(f"{'/'.join(v)}{list(sh)}" for sh, v in by_shape.items())
        outd = ", ".join(f"{dt}{list(sh)}" for dt, sh in outs)
        note = "" if prev_out is None else f"  <- untimed host conversions: {first} (first) + {again} x {reps - 1} (repeats)"
        rows.append(f"  {piece:28s} x{reps:>3d}  partitions {len(parts)}  in: {desc}  out: {outd}{note}")
        prev_out = outs
    return rows, per_pass


def pieces_total(tag: str, macs: int):
    f = WORK / "cells" / f"{tag}__Z{macs}__pieces.tsv"
    if not f.exists():
        return None
    row = next((l.split("\t") for l in f.read_text().splitlines() if l.startswith("TOTAL")), None)
    return float(row[4]) if row else None


def whole_total(tag: str, macs: int):
    f = WORK / "cells" / f"{tag}_whole__Z{macs}" / "result.tsv"
    if not f.exists() or not f.read_text().strip():
        return None
    cols = f.read_text().split("\t")  # tag macs n_partitions npu_active npu_total ms
    return float(cols[5]), int(cols[2])


def run_whole(tag: str, macs_list: list[int]) -> None:
    mode, argv = parse_lower_args((WORK / "lower" / tag / "lower.args").read_text())
    assert mode == "pieces", f"{tag}: lower.args mode is {mode}, expected pieces"
    procs = [subprocess.Popen(["bash", str(ROOT / "compile_and_profile.sh"), f"{tag}_whole", str(c), "--whole", *argv],
                              cwd=ROOT, stdout=open(WORK / "cells" / f"{tag}_whole__Z{c}.audit.log", "w"),
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
             for c in macs_list if whole_total(tag, c) is None]
    for p in procs:
        p.wait()


def main(tags: list[str], macs_list: list[int], run: bool) -> None:
    for tag in tags:
        print(f"== {tag}")
        rows, per_pass = piece_check(tag)
        print("\n".join(rows))
        print(f"  untimed host-side conversions per forward pass: {per_pass}")
        if run:
            run_whole(tag, macs_list)
        for c in macs_list:
            p, w = pieces_total(tag, c), whole_total(tag, c)
            if p is None:
                print(f"  Z{c:<5d} pieces: missing"); continue
            if w is None:
                print(f"  Z{c:<5d} pieces {p:10.3f} ms   whole: not measured (--run)"); continue
            ms, nparts = w
            print(f"  Z{c:<5d} pieces {p:10.3f} ms   whole {ms:10.3f} ms ({nparts} partitions)   "
                  f"pieces - whole {p - ms:+8.3f} ms ({100 * (p / ms - 1):+.2f} %)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tags", nargs="+", help="tags with piece results under $U85_WORK (lower/<tag>, cells/<tag>__Z*)")
    ap.add_argument("--macs", type=int, nargs="+", default=[256, 512, 1024, 2048], choices=sorted(CLOCK_HZ))
    ap.add_argument("--run", action="store_true", help="lower and profile the missing whole graphs")
    a = ap.parse_args()
    main(a.tags, a.macs, a.run)
