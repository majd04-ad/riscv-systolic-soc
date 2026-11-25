#!/usr/bin/env python3
"""Check an Ibex benchmark log against the host reference and print tables.

    compare.py <host bench.log> <ibex bench.log>

Both runs must report PASSED and contain the same, non-empty set of BENCH
lines, and every Ibex output hash must equal the host one. Exits non-zero
otherwise.
"""
import re
import sys

LINE = re.compile(r"BENCH (\S+) (\S+) (\S+) cycles=(\d+) instret=(\d+) "
                  r"macs=(\d+) hash=([0-9a-f]{8})")


def parse(path):
    rows = {}
    passed = False
    with open(path) as f:
        for line in f:
            m = LINE.match(line)
            if m:
                kind, name, phase = m.group(1, 2, 3)
                rows[(kind, name, phase)] = {
                    "cycles": int(m.group(4)),
                    "instret": int(m.group(5)),
                    "macs": int(m.group(6)),
                    "hash": m.group(7),
                }
            elif line.startswith("PASSED"):
                passed = True
    return rows, passed


def gemm_table(rows):
    out = ["| GEMM (M x N x K) | MACs | plain cycles | cycles/MAC | CPI | "
           "tiled (4x4 SW) cycles | tiled / plain |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    names = [k[1] for k in rows if k[0] == "gemm" and k[2] == "plain"]
    for name in names:
        p, t = rows[("gemm", name, "plain")], rows[("gemm", name, "tiled")]
        out.append(f"| {name.replace('x', ' x ')} | {p['macs']:,} | "
                   f"{p['cycles']:,} | {p['cycles'] / p['macs']:.2f} | "
                   f"{p['cycles'] / p['instret']:.2f} | {t['cycles']:,} | "
                   f"{t['cycles'] / p['cycles']:.2f} |")
    return "\n".join(out)


def conv_table(rows):
    out = ["| Layer | MACs | im2col cycles | GEMM cycles | total cycles | "
           "im2col share | cycles/MAC | direct conv cycles |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    names = [k[1] for k in rows if k[0] == "conv" and k[2] == "total"]
    for name in names:
        c, g, t, d = (rows[("conv", name, ph)]
                      for ph in ("im2col", "gemm", "total", "direct"))
        out.append(f"| {name} | {t['macs']:,} | {c['cycles']:,} | "
                   f"{g['cycles']:,} | {t['cycles']:,} | "
                   f"{100 * c['cycles'] / t['cycles']:.1f}% | "
                   f"{t['cycles'] / t['macs']:.2f} | {d['cycles']:,} |")
    return "\n".join(out)


def main():
    args = sys.argv[1:]
    if len(args) != 2:
        sys.exit(__doc__)
    host, host_passed = parse(args[0])
    ibex, passed = parse(args[1])

    errors = []
    if not host or not host_passed:
        sys.exit(f"{args[0]}: host reference is empty or did not report PASSED")
    if not passed:
        errors.append("Ibex run did not report PASSED")
    for key in sorted(ibex.keys() - host.keys()):
        errors.append(f"not in host reference: {' '.join(key)}")
    for key, ref in host.items():
        got = ibex.get(key)
        if got is None:
            errors.append(f"missing on Ibex: {' '.join(key)}")
        elif got["hash"] != ref["hash"]:
            errors.append(f"hash mismatch {' '.join(key)}: "
                          f"ibex {got['hash']} host {ref['hash']}")

    print(gemm_table(ibex))
    print()
    print(conv_table(ibex))
    print()
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print(f"OK: all {len(host)} outputs match the host reference")


if __name__ == "__main__":
    main()