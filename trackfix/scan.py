"""Compare the night you will mix against the other nights and list every glitch.

The playback machine plays the same files every night, so once another night is aligned
to a fraction of a sample the two recordings match almost exactly - typically to -40 to
-60 dB.  Anywhere they stop matching, one of them was damaged.

Per 60 s chunk of the reference night:
  1. split into music sections (playback is digitally silent between cues)
  2. find each section in every other night (1 kHz, whole recording) and keep the best
  3. align in 2 s blocks at 48 kHz, sub-sample, on the high frequencies, re-measured every
     block so clock drift and the operator's timing are both followed
  4. level-match per block and per channel - different recording gain or a fader ride
     on either night cannot matter
  5. flag (a) level dips: 0.5 ms windows one night >12 dB under the other
          (b) departures: 0.25 ms steps where the waveforms stop matching (residual
              within 15 dB of the music) - garbled stretches that barely dip
"""
import json, time
import numpy as np
from .audio import SR, Index, xlag, fshift, hf, runs

CHUNK, BLOCK, SEARCH = 60.0, 2.0, 0.020
DROP_DB, PRESENT_DB = -12.0, -25.0


def sections(X):
    """Music sections of a stereo chunk: above -80 dBFS, splitting on >= 1 s of silence."""
    w = int(0.05 * SR); m = X.shape[1] // w * w
    e = 10 * np.log10((X[:, :m] ** 2).mean(0).reshape(-1, w).mean(1) + 1e-20)
    on = e > -80; out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and (on[j + 1] or on[j + 1:j + 21].any()): j += 1
            if (j - i + 1) * 0.05 >= 1.0: out.append((i * w, (j + 1) * w))
            i = j + 1
        else:
            i += 1
    return out


_T = np.load(__import__("os").path.join(__import__("os").path.dirname(__file__), "dip_template.npy"))
_TQ = np.array([_T[i:i + 12].mean() for i in range(0, len(_T) - 12, 12)])   # 0.25 ms steps
_TZ, _BOT = _TQ - _TQ.mean(), int(np.argmin(_TQ))


def shape_hits(X, r_min=0.85, depth_max=-20, floor=-70):
    """Soft-mute dips found from their shape alone - no other night needed.  Validated on
    the show this was built on: 44 of 55 known dips, and no false alarms across nine
    cues, including a percussive number full of note gaps as deep as the dips.  Returns
    sample indices of each dip's bottom."""
    st = 12; m = X.shape[1] // st * st; n = len(_TQ)
    env = 10 * np.log10((X[:, :m] ** 2).mean(0).reshape(-1, st).mean(1) + 1e-20)
    if len(env) <= n: return []
    num = np.correlate(env, _TZ, "valid")
    c1 = np.concatenate([[0], np.cumsum(env)]); c2 = np.concatenate([[0], np.cumsum(env ** 2)])
    s1 = c1[n:] - c1[:-n]; s2 = c2[n:] - c2[:-n]
    r = num / (np.sqrt(np.maximum(s2 - s1 ** 2 / n, 1e-9)) * np.sqrt((_TZ ** 2).sum()))
    hits = []
    for k in np.flatnonzero(r >= r_min)[np.argsort(-r[r >= r_min])]:
        seg = env[k:k + n]; fl = np.median(np.r_[seg[:8], seg[-8:]])
        if fl < floor or seg[_BOT - 3:_BOT + 4].min() - fl > depth_max: continue
        if any(abs(k - h) < 80 for h in hits): continue                       # 20 ms apart
        hits.append(k)
    return sorted((k + _BOT) * st for k in hits)


def compare(X, Z):
    """Flags in stereo block X (reference) against aligned, level-matched Z (other night)."""
    ev = []
    W = 24
    x1, x2 = (X ** 2).sum(0), (Z ** 2).sum(0)
    m = len(x1) // W * W
    e1 = 10 * np.log10(x1[:m].reshape(-1, W).mean(1) + 1e-20)
    e2 = 10 * np.log10(x2[:m].reshape(-1, W).mean(1) + 1e-20)
    for who, lo, hi, med in (("reference", e1, e2, np.median(e2)), ("other", e2, e1, np.median(e1))):
        idx = np.flatnonzero((lo - hi < DROP_DB) & (hi > med + PRESENT_DB))
        for g in runs(idx, 4):
            ev.append({"who": who, "kind": "dip", "i": int(g[0] * W), "ms": (g[-1] - g[0] + 1) * W / SR * 1000,
                       "depth": float((lo - hi)[g].min())})
    st = 12; k = X.shape[1] // st
    Xs, Zs = X[:, :k * st].reshape(2, k, st), Z[:, :k * st].reshape(2, k, st)
    ex, ez, er = (Xs ** 2).sum((0, 2)), (Zs ** 2).sum((0, 2)), ((Xs - Zs) ** 2).sum((0, 2))
    big = np.maximum(ex, ez)
    dep = (er > 10 ** (-15 / 10) * big) & (big > np.median(big) * 10 ** (-25 / 10))
    for g in runs(np.flatnonzero(dep), 8):
        if len(g) < 2: continue
        who = "reference" if ex[g].sum() < ez[g].sum() else "other"
        ev.append({"who": who, "kind": "departure", "i": int(g[0] * st), "ms": (g[-1] - g[0] + 1) * st / SR * 1000})
    res = ((X - Z) ** 2).sum()
    null = 10 * np.log10(res / (x1.sum() + 1e-20) + 1e-20)
    return ev, null


def scan(ref, others, out_path, start=0.0, end=None, log=print):
    t0 = time.time()
    idx = [Index(o) for o in others]
    log(f"indexed {len(others)} other night(s) in {time.time()-t0:.0f}s")
    end = end or ref.duration()
    events, stats = [], {"blocks": 0, "unaligned_music": 0, "nulls": []}
    aligned = []                                            # (start s, end s, other night, offset) per block
    c0 = start
    while c0 < end:
        d = min(CHUNK, end - c0)
        X = ref.read(c0, d)
        for s0, s1 in sections(X):
            nd = min(8.0, (s1 - s0) / SR); mid = (s0 + s1) / 2 / SR
            needle = ref.read(c0 + max(s0 / SR, mid - nd / 2), nd, sr=1000)[0]
            cands = []
            for k, ix in enumerate(idx):
                for pos, r in ix.find(needle, top=1):
                    cands.append((r, k, pos - (c0 + max(s0 / SR, mid - nd / 2))))
            if not cands: continue
            r, k, off = max(cands)
            other = others[k]
            b = s0
            while b < s1:
                e = min(s1, b + int(BLOCK * SR))
                if e - b < int(0.2 * SR): break
                t2 = c0 + b / SR + off - SEARCH; dd = (e - b) / SR + 2 * SEARCH
                Y = other.read(t2, dd)
                x = hf(X[:, b:e].sum(0)); y = hf(Y.sum(0))
                lvl = 10 * np.log10((X[:, b:e] ** 2).mean() + 1e-20)
                if len(y) < len(x) + 4: break
                lag, rr = xlag(x, y)
                if rr < 0.5 and lvl > -55:                     # re-find it outright
                    hits = idx[k].find(ref.read(c0 + b / SR, min(4.0, (s1 - b) / SR), sr=1000)[0])
                    if hits:
                        off = hits[0][0] - (c0 + b / SR)
                        t2 = c0 + b / SR + off - SEARCH
                        Y = other.read(t2, dd); y = hf(Y.sum(0))
                        if len(y) >= len(x) + 4: lag, rr = xlag(x, y)
                if rr < 0.5:
                    if lvl > -55: stats["unaligned_music"] += 1
                    b = e; continue
                ki, fr = int(np.floor(lag)), lag - np.floor(lag)
                Z = np.stack([fshift(Y[c, ki:ki + (e - b) + 1], -fr)[:e - b] if Y.shape[1] >= ki + (e - b) + 1
                              else np.pad(Y[c, ki:], (0, max(0, ki + (e - b) - Y.shape[1])))[:e - b] for c in (0, 1)])
                Z = np.stack([Z[c] * (np.dot(X[c, b:e], Z[c]) / (np.dot(Z[c], Z[c]) + 1e-20)) for c in (0, 1)])
                ev, null = compare(X[:, b:e], Z)
                stats["blocks"] += 1; stats["nulls"].append(null)
                o = (t2 * SR + ki + fr) - (c0 * SR + b)               # other-night sample index minus reference index
                aligned.append((c0 + b / SR, c0 + e / SR, k, o))
                for v in ev:
                    v.update({"t": c0 + (b + v.pop("i")) / SR, "other": k, "offset": o, "null": null})
                    events.append(v)
                off = off + (lag - SEARCH * SR) / SR
                b = e
        # soft-mutes found by shape alone - catches dips where no other night lines up
        for h in shape_hits(X):
            t = c0 + h / SR
            if not (c0 <= t < c0 + d): continue
            if any(v["who"] == "reference" and v["kind"] == "dip" and abs(v["t"] - (t - 0.0065)) < 0.02 for v in events):
                continue
            blk = [a for a in aligned if a[0] - 0.5 <= t <= a[1] + 0.5]
            events.append({"who": "reference", "kind": "dip", "source": "shape", "t": t - 0.0065, "ms": 8.0,
                           "other": blk[0][2] if blk else None, "offset": blk[0][3] if blk else None, "null": None})
        mine = sum(1 for v in events if v["who"] == "reference")
        log(f"  {c0/60:6.1f} min   reference glitches so far {mine:>4}   blocks {stats['blocks']}"
            f"   music not found in another night {stats['unaligned_music']}")
        c0 += CHUNK
    stats["null_median"] = float(np.median(stats["nulls"])) if stats["nulls"] else None
    del stats["nulls"]
    json.dump({"events": events, "stats": stats,
               "reference": {"left": ref.left, "right": ref.right, "stereo": ref.stereo},
               "others": [{"left": o.left, "right": o.right, "stereo": o.stereo} for o in others]},
              open(out_path, "w"), indent=1)
    log(f"done in {(time.time()-t0)/60:.0f} min -> {out_path}")
    return events
