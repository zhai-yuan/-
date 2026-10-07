#!/usr/bin/env python3
"""Upload this release directory to the Linux build server and run make targets.

Why remote: the submitted artifact must be the artifact that was validated, and it is
a Linux `-fPIC -shared` object built by the contest clang.  This script copies the
self-contained release tree to the build box, runs the requested make targets there,
and prints the output.  It never prints credentials, never uploads a model to the
contest, and never starts an official game.

Connection details come from `<credentials_dir>/.goldrush_buildserver.json`, resolved
through `scripts/project_paths.py`, i.e. from OUTSIDE the repository.

Usage:
  python release/g14/validation/remote_make.py                                 # verify
  python release/g14/validation/remote_make.py --targets verify elf
  python release/g14/validation/remote_make.py --targets equiv equiv26 --rounds 200000
  python release/g14/validation/remote_make.py --targets bench
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
ROOT = RELEASE.parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_server  # noqa: E402

UPLOAD = [
    "Makefile",
    "src/player_v7.cpp",
    "src/game_api.h",
    "build.sh",
    "bin/player_norev_g14_pairemit_c74bea0c.so",
    "validation/player_norev_g12_raw_fused_b984d312.so",
    "validation/player_norev_g26_pair_maskedprev_dc18889f.so",
    "validation/so_equiv.cpp",
    "validation/rank1_rounds.h",
    "validation/bench_so.cpp",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=["verify"],
                    choices=["all", "verify", "elf", "equiv", "equiv26", "bench",
                             "toolchain", "clean"])
    ap.add_argument("--rounds", type=int, default=200_000,
                    help="random rounds for the equiv targets")
    ap.add_argument("--remote-name", default="release_g14")
    args = ap.parse_args()

    c = build_server.creds()
    if not build_server.check_key(c):
        print("build server key authentication is unavailable")
        return 2
    remote = c["remote_workdir"].rstrip("/") + "/" + args.remote_name
    target = build_server.target(c)

    build_server.ssh(
        c, f"rm -rf {remote} && mkdir -p {remote}/src {remote}/bin {remote}/validation")
    for rel in UPLOAD:
        local = RELEASE / rel
        if not local.is_file():
            raise SystemExit(f"missing {local}")
        subprocess.run(["scp", "-o", "BatchMode=yes", "-q", str(local),
                        f"{target}:{remote}/{rel}"], check=True)

    goals = " ".join(args.targets)
    script = (f"set -o pipefail; cd {remote} && "
              f"echo '--- toolchain ---' && clang++ --version | head -2 && "
              f"ld --version | head -1; "
              f"echo '--- make {goals} ---' && "
              f"make {goals} ROUNDS={args.rounds}")
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
         "-o", "StrictHostKeyChecking=accept-new", target, script],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"remote build dir: {remote}")
    print(r.stdout)
    if r.stderr.strip():
        print("stderr:", r.stderr.strip(), file=sys.stderr)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
