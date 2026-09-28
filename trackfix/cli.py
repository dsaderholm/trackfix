"""trackfix scan | repair | verify | all  <show.toml>"""
import argparse, json, os, sys, tomllib
import numpy as np
from .audio import SR, Night
from . import scan as scan_mod, repair as repair_mod


def load(path):
    cfg = tomllib.loads(open(path, encoding="utf-8").read())
    def night(d, name):
        return Night(left=d.get("left"), right=d.get("right"), stereo=d.get("stereo"), name=d.get("name", name))
    ref = night(cfg["reference"], "reference")
    others = [night(o, f"night {k + 2}") for k, o in enumerate(cfg.get("other", []))]
    if not others:
        print("no [[other]] nights: only soft-mute dips found by their shape can be repaired, by\n"
              "turning them back up - and garbled stretches cannot be found at all.  Record more nights.")
    out = cfg.get("output", {}).get("dir") or os.path.join(os.path.dirname(ref.path), "trackfix")
    os.makedirs(out, exist_ok=True)
    return ref, others, out


def tc(s):
    h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"


def paths(out):
    return (os.path.join(out, "scan.json"), os.path.join(out, "Tracks L (repaired).wav"),
            os.path.join(out, "Tracks R (repaired).wav"), os.path.join(out, "repairs.json"),
            os.path.join(out, "repairs.md"), os.path.join(out, "verify.json"))


def cmd_scan(ref, others, out, a):
    scan_mod.scan(ref, others, paths(out)[0], start=a.start or 0.0, end=a.end)


def cmd_repair(ref, others, out, a):
    sp, pl, pr, rj, rm, _ = paths(out)
    if not os.path.exists(sp): sys.exit("run `scan` first")
    notes = repair_mod.repair(ref, others, sp, pl, pr)
    json.dump(notes, open(rj, "w"), indent=1)
    done = [n for n in notes if n["status"] == "repaired"]
    left = [n for n in notes if n["status"].startswith("LEFT")]
    with open(rm, "w", encoding="utf-8") as f:
        f.write(f"# trackfix repairs\n\n{len(done)} repaired, {len(left)} left for checking by ear.\n\n")
        f.write("Times are from the start of the reference recording.\n\n")
        if left:
            f.write("## Check these by ear\n\n")
            for n in left: f.write(f"- {tc(n['t'])}  ({n['ms']:.1f} ms {n['kind']})\n")
            f.write("\n")
        f.write("## Repaired\n\n| time | span | source | fit |\n|---|---|---|---|\n")
        for n in done:
            span = (n.get("to_s", n["t"]) - n.get("from_s", n["t"])) * 1000
            fit = f"{n['fit_db']:.0f} dB" if "fit_db" in n else "-"
            f.write(f"| {tc(n['t'])} | {span:.0f} ms | {n['source']} | {fit} |\n")
    print(f"\n{len(done)} repaired, {len(left)} left for checking by ear -> {rm}")


def cmd_verify(ref, others, out, a):
    """Re-scan the repaired recording against the other nights.  Anything left that shows a
    real dip in its own level is listed; the rest is the other nights' damage or noise."""
    sp, pl, pr, rj, rm, vj = paths(out)
    fixed = Night(left=pl, right=pr, name="repaired")
    ev = scan_mod.scan(fixed, others, vj, start=a.start or 0.0, end=a.end)
    # what is left that still looks like a soft-mute - found by shape, or flagged against
    # another night and soft-mute shaped.  Gaps between notes do not count.
    real = []
    for e in ev:
        if e["who"] != "reference" or e["kind"] != "dip" or e["ms"] > 40: continue
        X = fixed.read(e["t"] - 0.05, 0.1 + e["ms"] / 1000)
        r, d = repair_mod.soft_mute_shape(X, int(0.05 * SR), e["ms"])
        if e.get("source") == "shape" or (r >= 0.8 and d <= -20): real.append(e)
    print(f"\nverify: {len(real)} soft-mute dips still present in the repaired recording")
    for e in real: print(f"   {tc(e['t'])}  {e['ms']:.1f} ms")


def main():
    ap = argparse.ArgumentParser(prog="trackfix", description=__doc__)
    ap.add_argument("command", choices=["scan", "repair", "verify", "all"])
    ap.add_argument("config")
    ap.add_argument("--start", type=float, help="seconds into the reference (testing)")
    ap.add_argument("--end", type=float, help="seconds into the reference (testing)")
    a = ap.parse_args()
    ref, others, out = load(a.config)
    for c in (["scan", "repair", "verify"] if a.command == "all" else [a.command]):
        print(f"\n=== {c} ===")
        {"scan": cmd_scan, "repair": cmd_repair, "verify": cmd_verify}[c](ref, others, out, a)


if __name__ == "__main__":
    main()
