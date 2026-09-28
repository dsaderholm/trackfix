"""Reading, aligning and comparing recordings.  Everything goes through ffmpeg."""
import subprocess
import numpy as np

SR = 48000


def read(path, t0, d, sr=SR, ch=None):
    """Mono float64 from `path`, `d` seconds from `t0`.  ch picks a channel of a stereo file."""
    af = ["-af", f"pan=mono|c0=c{ch}"] if ch is not None else ["-ac", "1"]
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(0.0, t0):.6f}", "-t", f"{d:.6f}",
                        "-i", path, *af, "-ar", str(sr), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(r, np.float32).astype(np.float64)


def duration(path):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True).stdout
    return float(o.strip())


class Night:
    """One night's tracks recording: a left and right mono file (as a multitrack recorder
    writes them) or one stereo file."""

    def __init__(self, left=None, right=None, stereo=None, name=""):
        self.left, self.right, self.stereo, self.name = left, right, stereo, name
        self.path = left or stereo

    def read(self, t0, d, sr=SR):
        if self.stereo:
            return np.stack([read(self.stereo, t0, d, sr, ch=0), read(self.stereo, t0, d, sr, ch=1)])
        L, R = read(self.left, t0, d, sr), read(self.right, t0, d, sr)
        n = min(len(L), len(R))
        return np.stack([L[:n], R[:n]])

    def duration(self):
        return duration(self.path)


class Index:
    """A whole recording at 1 kHz with its FFT cached, for finding a passage anywhere in it."""

    def __init__(self, night):
        h = read(night.left or night.stereo, 0, 1e6, sr=1000, ch=None if night.left else 0)
        self.h = h
        self.n = 1 << int(np.ceil(np.log2(len(h) + 20000)))
        self.H = np.fft.rfft(h, self.n)
        self.c = np.concatenate([[0.0], np.cumsum(h ** 2)])

    def find(self, needle, top=1):
        """Best `top` (time_s, r) matches for a 1 kHz needle."""
        m = len(needle)
        if m < 100 or m >= len(self.h): return []
        cc = np.fft.irfft(self.H * np.conj(np.fft.rfft(needle, self.n)), self.n)[:len(self.h) - m]
        en = np.sqrt((self.c[m:len(self.h)] - self.c[:len(self.h) - m]) * (needle ** 2).sum())
        r = cc / (en + 1e-9 * en.max())
        out = []
        for _ in range(top):
            k = int(np.argmax(r))
            if r[k] <= 0: break
            out.append((k / 1000.0, float(r[k])))
            r[max(0, k - 2000):k + 2000] = -1
        return out


def xlag(x, y):
    """Where x sits inside the longer y: (fractional lag in samples, normalised correlation)."""
    n = 1 << int(np.ceil(np.log2(len(y) + len(x))))
    cc = np.fft.irfft(np.fft.rfft(y, n) * np.conj(np.fft.rfft(x, n)), n)[:len(y) - len(x) + 1]
    k = int(np.argmax(cc))
    r = float(cc[k] / (np.sqrt((x ** 2).sum() * (y[k:k + len(x)] ** 2).sum()) + 1e-20))
    if 0 < k < len(cc) - 1:
        a, b, c = cc[k - 1], cc[k], cc[k + 1]
        d = a - 2 * b + c
        return k + (0.5 * (a - c) / d if d else 0.0), r
    return float(k), r


def fshift(y, s):
    """y delayed by a fractional s samples, same length."""
    n = len(y); f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(y) * np.exp(-2j * np.pi * f * s), n)


def hf(x):
    """Second difference - a steep high-pass.  A held note repeats every pitch cycle, so a
    broadband match has several equally good answers one cycle apart; the high end
    (breath, reverb, detail) does not repeat."""
    return np.diff(x, 2)


def runs(idx, gap):
    """Split sorted indices into runs, merging gaps up to `gap`."""
    if not len(idx): return []
    return np.split(idx, np.flatnonzero(np.diff(idx) > gap) + 1)


def level_db(x):
    return 10 * np.log10((x ** 2).mean() + 1e-20)
