"""Repair the reference night's glitches, writing a new recording that is bit-identical
everywhere except the repaired spots.  Same length, same format - a drop-in.

For each glitch the scan found in the reference night, in order of preference:

  1. THE OTHER NIGHT AT THE SAME SPOT.  Aligned there on the high frequencies to a
     fraction of a sample, widened to cover the whole disturbance (a flag can be a sliver
     of a longer one), level-matched per channel to the 60 ms either side, crossfaded
     over 1.5 ms.  Refused unless it fits the music on both sides to -12 dB or better
     and is not dipping itself.
  2. ANOTHER COPY OF THE SAME MUSIC anywhere in any night (scene-change music gets
     reused).  Both sides must line up, with identical timing.  Same tests.
  3. TURN THE DIP BACK UP.  Playback soft-mutes turn the music down without stopping it,
     by the same curve every time; the measured template is fitted and divided out.
     Only for soft-mute-shaped dips with a real dip in the reference's own level, and
     only if the level comes out smooth - otherwise the spot is left alone and listed.

Nothing is inserted or removed: timing either side of these glitches is unchanged to a
hundredth of a sample, so length and sync with every other track are untouched.
"""
import json, os, subprocess
import numpy as np
from .audio import SR, Night, Index, read, xlag, fshift, hf, runs

PAD, XF = 0.003, 0.0015
M = int(0.06 * SR)
MARGIN = 2.0                                  # seconds of context read around each chunk
TEMPLATE = np.load(os.path.join(os.path.dirname(__file__), "dip_template.npy"))   # dB, bottom at 600
BOTTOM = 600


# ------------------------------------------------------------------ helpers

def aligned(other, i0, i1, o):
    """The other night's audio for reference samples [i0, i1), o = its index minus ours."""
    L = i1 - i0; base = int(np.floor(i0 + o)); fr = (i0 + o) - base
    seg = other.read(base / SR, (L + 4) / SR)
    if seg.shape[1] < L + 1: return None
    return np.stack([fshift(seg[c, :L + 1], -fr)[:L] for c in (0, 1)])


def local_offset(X, x0, c0, c1, other, guess_o):
    """Other-night index minus reference index, from clean music in X[:, c0:c1] (X starts at x0)."""
    if c0 < 0 or c1 > X.shape[1]: return None
    x = hf(X[:, c0:c1].sum(0))
    t2 = (x0 + c0 + guess_o) / SR - 0.06
    y = hf(other.read(t2, (c1 - c0) / SR + 0.12).sum(0))
    if len(y) < len(x) + 4: return None
    lag, r = xlag(x, y)
    return t2 * SR + lag - (x0 + c0) if r > 0.6 else None


def disturbance(X, x0, A, B, other, o):
    """Widen [A, B) (indices into X) to the whole stretch where X departs from the copy."""
    R = int(0.025 * SR); s0, s1 = A - R, B + R
    if s0 < 0 or s1 > X.shape[1]: return A, B
    z = aligned(other, x0 + s0, x0 + s1, o)
    if z is None: return A, B
    x = X[:, s0:s1]; L = s1 - s0; ed = np.r_[0:480, L - 480:L]
    z = z * (np.dot(x[:, ed].ravel(), z[:, ed].ravel()) / (np.dot(z[:, ed].ravel(), z[:, ed].ravel()) + 1e-20))
    st = 12; k = L // st
    xs, zs = x[:, :k * st].reshape(2, k, st), z[:, :k * st].reshape(2, k, st)
    ex, ez, er = (xs ** 2).sum((0, 2)), (zs ** 2).sum((0, 2)), ((xs - zs) ** 2).sum((0, 2))
    bad = (er > 10 ** (-15 / 10) * np.maximum(ex, ez)) & (np.maximum(ex, ez) > 1e-12)
    a_i, b_i = (A - s0) // st, (B - s0) // st; lo, hi = a_i, b_i
    for r_ in runs(np.flatnonzero(bad), 8):
        if r_[-1] >= a_i - 8 and r_[0] <= b_i + 8: lo, hi = min(lo, r_[0]), max(hi, r_[-1] + 1)
    lo, hi = max(0, lo - 4), min(k, hi + 4)
    nA, nB = s0 + lo * st, s0 + hi * st
    return (nA, nB) if nB - nA <= int(0.045 * SR) else (A, B)


def place(Y, X, z, A, B):
    """z covers [A-M, B+M).  Level-match on the anchors, check, crossfade [A, B) into Y.
    Returns (ok, fit_db)."""
    ref = np.concatenate([X[:, A - M:A], X[:, B:B + M]], 1)
    zz = np.concatenate([z[:, :M], z[:, -M:]], 1)
    g = [np.dot(ref[c], zz[c]) / (np.dot(zz[c], zz[c]) + 1e-20) for c in (0, 1)]
    z = np.stack([g[0] * z[0], g[1] * z[1]]); zz = np.concatenate([z[:, :M], z[:, -M:]], 1)
    fit = 10 * np.log10(((ref - zz) ** 2).mean() / ((ref ** 2).mean() + 1e-20) + 1e-20)
    core = z[:, M:-M]; inner = core[:, M // 2:-M // 2] if core.shape[1] > M else core
    if fit > -12 or 10 * np.log10((inner ** 2).mean() + 1e-20) < 10 * np.log10((zz ** 2).mean() + 1e-20) - 12:
        return False, fit
    x = int(XF * SR); w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, x))
    mix = np.ones(B - A); mix[:x] = w; mix[-x:] = w[::-1]
    Y[:, A:B] = Y[:, A:B] * (1 - mix) + core * mix
    return True, fit


def envelope_ms(X, i, ms, before=0.015, after=0.015):
    i0, i1 = i - int(before * SR), i + int((ms / 1000 + after) * SR)
    if i0 < 0 or i1 > X.shape[1]: return None, None
    s = X[:, i0:i1]; m = s.shape[1] // 48 * 48
    e = 10 * np.log10((s[:, :m] ** 2).mean(0).reshape(-1, 48).mean(1) + 1e-20)
    return e, np.median(np.r_[e[:int(before * 1000) - 4], e[-(int(after * 1000) - 4):]])


def real_dip(X, i, ms):
    env, fl = envelope_ms(X, i, ms)
    return env is not None and (env[11:-11].min() - fl) <= -8


def smooth(X, i, ms):
    env, fl = envelope_ms(X, i, ms)
    if env is None: return False
    inner = env[13:-13]
    return (inner.min() - fl) > -10 and (inner.max() - fl) < 6


def gain_curve(n, pos, scale):
    idx = np.arange(n) - pos + BOTTOM
    return 10 ** (np.interp(idx, np.arange(len(TEMPLATE)), TEMPLATE, left=0.0, right=0.0) * scale / 20)


def unduck(Y, i, ms):
    """Fit the soft-mute template around sample i of Y and divide it out, in place.
    Tries a focused fit, then a wide one; keeps it only if the level comes out smooth."""
    a0, b0 = i - int(0.030 * SR), i + int(0.040 * SR)
    if a0 < 0 or b0 > Y.shape[1]: return None
    x = Y[:, a0:b0].copy()
    guess = (i - a0) + (ms / 1000 - 0.002) * SR
    for positions, scales in ((np.arange(guess - 0.004 * SR, guess + 0.004 * SR, 2.0), np.linspace(0.4, 1.4, 26)),
                              (np.arange(0.018 * SR, 0.044 * SR, 12), np.linspace(0.2, 1.2, 21))):
        best = None
        for pos in positions:
            for sc in scales:
                Y[:, a0:b0] = x / gain_curve(x.shape[1], pos, sc)
                env, fl = envelope_ms(Y, i, ms)
                if env is None: continue
                inner = env[11:-11]; d = max(inner.max() - fl, fl - inner.min())
                if best is None or d < best[0]: best = (d, pos, sc)
        if best is None: continue
        Y[:, a0:b0] = x / gain_curve(x.shape[1], best[1], best[2])
        if smooth(Y, i, ms): return {"bottom_s": (a0 + best[1]) / SR, "depth_scale": float(best[2])}
    Y[:, a0:b0] = x
    return None


# ------------------------------------------------------------------ the repair

def repair(ref, others, scan_path, out_left, out_right, log=print):
    rep = json.load(open(scan_path))
    # Level dips drive the repair; each patch is widened to the whole disturbance.  Waveform
    # departures alone are not targets - on the show this was built on, 42 of 50 isolated
    # ones moved less than 6 dB (the nights' analog noise disagreeing) and the deep ones
    # were the other night's damage.  They are used by `verify`.
    evs = sorted([e for e in rep["events"] if e["who"] == "reference" and e["kind"] == "dip"], key=lambda e: e["t"])
    theirs = [e for e in rep["events"] if e["who"] == "other" and e["kind"] == "dip"]
    total = ref.duration(); n_total = int(round(total * SR))
    idx = None
    notes = []
    writers = [subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                                 "-i", "-", "-c:a", "pcm_s24le", p], stdin=subprocess.PIPE) for p in (out_left, out_right)]
    written = 0
    c0 = 0.0
    while written < n_total:
        d = min(60.0, total - c0)
        x0 = max(0, int(round((c0 - MARGIN) * SR)))
        X = ref.read(x0 / SR, d + 2 * MARGIN)
        Y = X.copy()
        lo, hi = int(round(c0 * SR)) - x0, int(round((c0 + d) * SR)) - x0
        for e in evs:
            i = int(round(e["t"] * SR)) - x0
            if not (lo <= i < hi): continue
            if any(n["status"] == "repaired" and n["from_s"] - 0.001 <= e["t"] <= n["to_s"] + 0.001 for n in notes):
                continue
            note = {"t": e["t"], "kind": e["kind"], "ms": e["ms"]}
            if e["ms"] > 40:
                notes.append({**note, "status": "edge - the nights differ here, not a glitch"}); continue
            a, b = i - int(PAD * SR), i + int((e["ms"] / 1000 + PAD) * SR)
            if a - M - int(0.3 * SR) < 0 or b + M + int(0.3 * SR) > X.shape[1]:
                notes.append({**note, "status": "LEFT AS IS - too close to the recording's edge"}); continue
            other = others[e["other"]]
            ob = local_offset(X, x0, a - int(0.26 * SR), a - int(0.01 * SR), other, e["offset"])
            oa = local_offset(X, x0, b + int(0.01 * SR), b + int(0.26 * SR), other, e["offset"])
            done = False
            damaged = any(t["other"] == e["other"] and abs(t["t"] - e["t"]) < 0.010 for t in theirs)
            if (ob is not None or oa is not None) and not damaged and (ob is None or oa is None or abs(ob - oa) < 2):
                o = float(np.mean([v for v in (ob, oa) if v is not None]))
                A, B = disturbance(X, x0, a, b, other, o)
                z = aligned(other, x0 + A - M, x0 + B + M, o)
                if z is not None:
                    ok, fit = place(Y, X, z, A, B)
                    if ok:
                        notes.append({**note, "status": "repaired", "source": f"{other.name or 'other night'} at the same spot",
                                      "fit_db": fit, "from_s": (x0 + A) / SR, "to_s": (x0 + B) / SR}); done = True
            if not done:
                if idx is None: idx = [Index(o_) for o_ in others]
                needle = read(ref.left or ref.stereo, e["t"] - 4.2, 4.0, sr=1000, ch=None if ref.left else 0)
                for k, (oth, ix) in enumerate(zip(others, idx)):
                    for pos, r in ix.find(needle, top=4):
                        if r < 0.5: continue
                        o_guess = (pos + 4.2 - e["t"]) * SR
                        o1 = local_offset(X, x0, a - int(0.26 * SR), a - int(0.01 * SR), oth, o_guess)
                        o2 = local_offset(X, x0, b + int(0.01 * SR), b + int(0.26 * SR), oth, o_guess)
                        if o1 is None or o2 is None or abs(o1 - o2) >= 2: continue
                        o = (o1 + o2) / 2
                        A, B = disturbance(X, x0, a, b, oth, o)
                        z = aligned(oth, x0 + A - M, x0 + B + M, o)
                        if z is None: continue
                        ok, fit = place(Y, X, z, A, B)
                        if ok:
                            notes.append({**note, "status": "repaired", "fit_db": fit,
                                          "source": f"copy of the same music in {oth.name or 'another night'} at {(x0 + A + o) / SR:.2f}s",
                                          "from_s": (x0 + A) / SR, "to_s": (x0 + B) / SR}); done = True; break
                    if done: break
            if not done:
                if not real_dip(X, i, e["ms"]):
                    notes.append({**note, "status": "edge - no dip in this night's own level (the other night differs)"})
                    continue
                res = unduck(Y, i, e["ms"])
                if res:
                    notes.append({**note, "status": "repaired", "source": "own audio turned back up (soft-mute template)",
                                  "from_s": e["t"] - 0.03, "to_s": e["t"] + 0.04, **res}); done = True
            if not done:
                notes.append({**note, "status": "LEFT AS IS - check by ear"})
        chunk = Y[:, lo:hi]
        if written + chunk.shape[1] > n_total: chunk = chunk[:, :n_total - written]
        for c in (0, 1):
            writers[c].stdin.write(chunk[c].astype(np.float32).tobytes())
        written += chunk.shape[1]
        rep_n = sum(1 for n in notes if n["status"] == "repaired")
        log(f"  {c0/60:6.1f} min   repaired so far {rep_n:>4}   left {sum(1 for n in notes if n['status'].startswith('LEFT')):>3}")
        c0 += d
    for w in writers:
        w.stdin.close(); w.wait()
    return notes
