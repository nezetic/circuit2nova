Circuit2nova
============

Introduction
------------

*circuit2nova* is an experimental project to convert Novation [Circuit](https://www.vintagesynth.com/novation/Circuit) (including [Tracks](https://novationmusic.com/products/circuit-tracks)) presets to the [UltraNova](https://www.vintagesynth.com/novation/ultranova) (and Mininova).

Both synths share a very similar engine, with the Circuit being both a simplified (less oscillators, envelopes, LFOs, filters and FX) and updated version (more modern sounding wavetables, macros).

The tool is a simple script, requiring only Python (>= 3.9).

*Disclaimer*: This is a toy project (not affiliated in any way with Novation)!
Conversion is based on my own reverse engineering of the UltraNova presets file format.
Use this tool at your own risk!


Goals
-----

As engines are different, presets will *never* sound exactly the same.
My goal was to use presets created for the Circuit as "templates" on my UltraNova.

But results are very often surprisingly close to the original preset, especially with a bit of fine tuning.


Usage
-----

To convert a Circuit preset to the UltraNova format:

```
$ ./circuit2nova.py single "Psy Bass.syx" "Psy Bass - UltraNova.syx"
```

It's also possible to convert multiple presets in one command:

```
$ ./circuit2nova.py batch "$HOME/PRESETS/CIRCUIT TRACKS" OUTDIR
```

If Circuit presets have meaningless names, it's possible to rename them automatically during the conversion using the `-r` option:

```
$ ./circuit2nova.py batch -r "$HOME/PRESETS/CIRCUIT TRACKS" OUTDIR
```

Generated presets are not assigned to any bank (*edit buffer*). They can be synced with the original Novation librarian or the amazing [KnobKraft Orm](https://github.com/christofmuc/KnobKraft-orm), a free, modern and cross-platform MIDI Sysex Librarian (compatible with the latest macOS).


Details
-------

Conversion is based on a *template* UltraNova preset, with filters and FX slots pre-configured to match the Circuit engine.

Apart from adapting the file format itself, the tool also modify the presets by remapping some values, including:

- Osc2 to Osc3, to preserve:
    - ring modulation compatibility;
    - ability to bypass Osc1 only (based on UltraNova *parallel2* routing);
- Chorus / EQ / Distortion to serial FX slots 1/2/3
    - with slots 4 and 5 mapped to delay / reverb;
- filter macro (if any) to modwheel;
- macros' destinations (**A** slots) to tweaks' assignments (less powerful but better than nothing).


Limitations
-----------

Oscillators' wavetables are very different. They are remapped automatically to sound *not too far off*.
But results might vary greatly depending on the wavetable.

As the UltraNova doesn't have macros, their modulations are *burned* during the conversion based on knob positions in the Circuit preset.
First macro destination (**A** slot) is assigned to corresponding UltraNova tweak for easy access. But this mechanism is way less powerful than the Circuit macro system that can control up to 4 parameters with various ranges.


Documentations
--------------

Conversion is done through a mapping of the respective preset formats.

Circuit Tracks is well documented, with both MIDI and presets specifications provided by [Novation](https://downloads.novationmusic.com/novation/circuit/circuit-tracks).

This is unfortunately not the case for the UltraNova, and the preset format was obtained through reverse engineering.
The process was mostly automated based on my own CC/NPM chart for the UltraNova, so probably not error-free.

- [Circuit Tracks Patch](https://docs.google.com/spreadsheets/d/1MgNMSnWRSUlFp8cW8Ld0qWe9cIHNAKExq_OS_-P_304)
- [UltraNova Patch](https://docs.google.com/spreadsheets/d/17GIJMeY8kT7Dybi3vs6D2L998KQcNJipquv5iKJojRQ)
- [Ultranova CC/NPM Chart](https://docs.google.com/spreadsheets/d/10qpeJyHA0gnSd1361xd2vCzAYNnDFRRcveovo5-_X5E)

