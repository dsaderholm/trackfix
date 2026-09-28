# trackfix

Repair playback glitches in a show's backing-track recording, using the other nights.

If a show's tracks are played from a computer (QLab, a DAW, anything) and you record
more than one performance, every night plays **the same files**. Line two nights up to
a fraction of a sample and they match almost perfectly - typically to 40-60 dB below the
music. Anywhere they stop matching, one of them was damaged, and the other one has the
real audio for that spot.

trackfix finds every such spot in the night you are going to mix, repairs it from the
other nights, and writes a repaired recording that is **bit-identical everywhere else** -
same length, same format, a drop-in for the original.

Built after a school production where the QLab playback machine soft-muted the tracks
for ~15 ms roughly once a minute, on every night, at random. Inaudible in the room;
obvious on a good car stereo.

## Use it the day after the show

Before cutting cues or mixing anything:

1. Copy `examples/show.toml` next to the show and fill in the paths: the night you will
   mix as `[reference]`, every other recorded night as an `[[other]]` block.
2. Run it:

   ```
   python -m trackfix all show.toml
   ```

   `all` = `scan`, then `repair`, then `verify`. Each can be run on its own.
3. Read `repairs.md` in the output folder. It lists every repair with its time and
   where the audio came from, and anything left for you to check by ear - spots that dip
   deep and look partly like a soft-mute, where it would rather ask than guess.
4. Cut your cues from `Tracks L (repaired).wav` / `Tracks R (repaired).wav`, not from
   the originals. Keep the originals.

A 3-hour show takes roughly an hour for `scan`, a similar time for `repair`, and another
hour for `verify`. Windows, Python 3.11+, numpy, and ffmpeg on the PATH.

## What it does

**scan** - the reference night, in 60 s chunks:

- splits into music sections (playback is digitally silent between cues, and the
  operator's timing differs night to night, so each section is placed separately)
- finds each section in every other night, keeps the best match
- aligns 2 s blocks at 48 kHz to a fraction of a sample, re-measured every block - on
  the high frequencies, because a held note repeats every pitch cycle and fools a
  broadband match by exactly one cycle
- level-matches per block and per channel, so a different recording gain or a fader
  ride on either night cannot matter
- flags level dips (0.5 ms windows, one night >12 dB under the other) and waveform
  departures (0.25 ms steps where the recordings stop matching)
- **also searches for the soft-mute's shape directly** in the reference, needing no other
  night: a slow ~11 ms slide down and a fast snap back. On the show this was built on it
  found 44 of 55 known dips with no false alarms, including in a percussive number whose
  gaps between notes dip just as deep. It catches dips where no other night lines up;
  the night-to-night comparison catches the ones that land on a note attack

**repair** - each dip in the reference, in order of preference:

1. **the other night at the same spot** - widened to cover the whole disturbance (a flag
   can be a sliver of a 15 ms glitch), level-matched to the 60 ms either side,
   crossfaded over 1.5 ms. Refused unless it fits the music on both sides to -12 dB or
   better and the other night is not damaged there itself
2. **another copy of the same music** anywhere in any night (scene-change music gets
   reused), both sides lining up with identical timing
3. **turning the dip back up** - playback soft-mutes turn the music down without stopping
   it, by the same curve every time (`dip_template.npy`, measured on 48 of them). Only
   where the reference's own level has **that shape** - a slow ~11 ms slide down and a
   fast snap back, at least 20 dB deep - and only if the level comes out smooth. A gap
   between notes can dip just as deep; "turning it back up" would fill it in and flatten
   the rhythm, so shape, not depth, is the test

Nothing is inserted or removed. On the show this was built for, timing either side of
every glitch was unchanged to a hundredth of a sample.

**verify** - scans the repaired recording against the other nights the same way and
lists anything that still dips.

## What it cannot do

- **One night only.** It still runs: soft-mutes are found by their shape and turned
  back up. But garbled stretches can't be found without a second night to compare
  against, and nothing can be patched from real audio. Record every night.
- **A glitch at the same moment on every night** - that is in the file, not the playback.
  Never seen, but it would be reported as nothing.
- **Different audio each night** - pre-show house music, a re-exported cue, a different
  playlist. Those sections are simply not matched and are left alone.

## Prevent it instead

Figure 53's guide *A Computer Prepares*
(qlab.app/docs/v5/general/preparing-your-mac): power connected, sleep off, QLab in
Show Mode, other apps closed, Wi-Fi/Bluetooth/iCloud off, Spotlight and Time Machine
off, uncompressed audio at the output's sample rate. Then play a long cue into the desk
for twenty minutes and look at the recording at 1 ms zoom. And ask for the final audio
files - then the recording is only a sync reference.

Keep recording every night regardless. It is what makes this repairable.

## License

MIT
