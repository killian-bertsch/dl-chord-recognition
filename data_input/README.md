# Input Data Directory

This directory is for your raw input data.

## Directory Structure

```
data_input/
├── 00_raw/          # Ableton exports (copy from original project)
└── 00_input/        # MusicXML + full audio (optional, for future use)
```

## Setup Instructions

### Copy from Original Project

If you have the original Deep Learning project:

```bash
# From the Chord_Recognition_Refactored directory
cp -r "../Deep Learning/data/00_raw/"* data_input/00_raw/
```

### Expected Structure for 00_raw

Each song should have its own directory:

```
00_raw/
├── beatrice/
│   ├── beatrice_chords        # Chord progression file
│   └── beatrice_clips/         # Audio slice exports
│       ├── Slice 1 [timestamp].wav
│       ├── Slice 2 [timestamp].wav
│       └── ...
├── blue_bossa/
│   ├── blue_bossa_chords
│   └── blue_bossa_clips/
└── autumn_leaves/
    ├── autumn_leaves_chords
    └── autumn_leaves_clips/
```

### Chord File Format

The `*_chords` file should contain one chord per line:

```
Fmaj7
Gbmaj7
Dm7 | Cm7          # Two chords in one bar (pipe-separated)
Bbm7
Am7b5 | A7
```

## Notes

- The preprocessing notebook will automatically detect and process all songs in `00_raw/`
- Duplicate slices from Ableton exports are automatically handled
- Bars with 2 chords (separated by `|`) are automatically split in half

## What Gets Processed

The `data_preprocessing.ipynb` notebook will:
1. Scan all song directories in `00_raw/`
2. Load chord progressions from `*_chords` files
3. Load audio slices from `*_clips/` directories
4. Deduplicate and organize all the data
5. Apply augmentation and feature extraction
6. Output CSV files to `../data_output/`
