# Chord Recognition - Refactored for Kaggle

A streamlined audio chord recognition project designed for Kaggle notebook training.

## Overview

This is a refactored version of the original Deep Learning chord recognition project, optimized for:
- **Simplified data pipeline** - Single Jupyter notebook for all preprocessing
- **Kaggle-friendly output** - Features exported as CSV files
- **No intermediate folders** - Direct: raw data → CSV features
- **Easy upload** - Upload CSV datasets to Kaggle and train there

## Project Structure

```
Chord_Recognition_Refactored/
├── README.md                   # This file
├── CLAUDE.md                   # Complete project documentation
├── requirements.txt            # Python dependencies
│
├── data_preprocessing.ipynb    # Main preprocessing notebook
│
├── data_input/                 # Input data directory
│   ├── 00_raw/                 # Ableton exports (copy from original project)
│   └── 00_input/               # MusicXML + full audio (optional)
│
└── data_output/                # Output CSV files
    ├── chroma_features.csv     # 12-bin chromagram features
    ├── hybrid_features.csv     # 36-bin hybrid features
    ├── cqt60_features.csv      # 60-bin CQT features
    └── cqt84_features.csv      # 84-bin CQT features
```

## Quick Start

### 1. Setup

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Input Data

**Option A: Copy from original project**
```bash
# Copy raw Ableton exports
cp -r "../Deep Learning/data/00_raw/"* data_input/00_raw/
```

**Option B: Use existing processed data**
- If you have the original `00_raw` folder with Ableton exports, copy it to `data_input/00_raw/`

### 3. Run Preprocessing

Open and run the Jupyter notebook:

```bash
jupyter notebook data_preprocessing.ipynb
```

The notebook will:
1. Load audio from `data_input/00_raw/`
2. Deduplicate and split bars
3. Apply pitch-shift augmentation (4x data increase)
4. Extract 4 types of audio features
5. Export everything to CSV files in `data_output/`

### 4. Upload to Kaggle

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Click "New Dataset"
3. Upload the CSV files from `data_output/`
4. Create a new notebook and start training!

## Features

### Input Formats Supported

1. **Ableton Exports** (`00_raw/`)
   - Bar-by-bar audio slices
   - Chord progression text file
   - Automatically handles duplicates and bar splitting

2. **MusicXML + Audio** (`00_input/`) - Future
   - Full-length recordings
   - Chord charts from iRealPro
   - Auto-segmentation

### Feature Types

| Feature Type | Dimensions | Description |
|--------------|------------|-------------|
| **chroma** | 12 × 87 = 1,044 | Octave-invariant pitch classes, best for root detection |
| **hybrid** | 36 × 87 = 3,132 | Low-CQT + Chroma stacked, best overall performance |
| **cqt60** | 60 × 87 = 5,220 | Reduced CQT, good balance |
| **cqt84** | 84 × 87 = 7,308 | Full-range CQT, most detailed |

All features are normalized to 87 time frames and exported as flattened arrays.

### CSV Format

Each CSV file contains:
- `id` - Sample identifier (e.g., "beatrice_bar01_shift+2")
- `label` - Full chord label (e.g., "F:maj")
- `root` - Root note only (e.g., "F")
- `quality` - Chord quality (e.g., "maj", "min", "dom")
- `song` - Song name
- `shift` - Pitch shift amount (-2, 0, 2, 4)
- `feature_0` to `feature_N` - Flattened feature array

## Data Augmentation

The preprocessing applies pitch-shift augmentation:
- **Shifts:** -2, 0, +2, +4 semitones (configurable)
- **Effect:** 4x dataset increase
- **Label transposition:** Chord labels are automatically transposed

Example:
```
Original: beatrice_bar01.wav → F:maj
Shift +2: beatrice_bar01_shift+2.wav → G:maj
```

## Advantages Over Original

| Original Project | Refactored Version |
|------------------|-------------------|
| 5 data directories | 1 input, 1 output |
| Multiple Python scripts | Single Jupyter notebook |
| .npy files (not portable) | CSV files (Kaggle-ready) |
| Local training | Cloud training on Kaggle |
| Complex pipeline | Streamlined workflow |

## Training (on Kaggle)

After uploading the CSV files to Kaggle:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Load data
df = pd.read_csv('../input/your-dataset/chroma_features.csv')

# Separate features and labels
X = df.filter(regex='feature_').values
y_root = df['root'].values
y_full = df['label'].values

# Group split (prevent data leakage from augmentation)
groups = df['id'].str.replace(r'_shift.*', '', regex=True)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X, y_root, groups))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y_root[train_idx], y_root[val_idx]

# Reshape for CNN: (samples, 1, n_bins, 87)
n_bins = 12  # for chroma
X_train = X_train.reshape(-1, 1, n_bins, 87)
X_val = X_val.reshape(-1, 1, n_bins, 87)

# Build and train your model...
```

## Dependencies

- Python 3.8+
- NumPy
- pandas
- librosa
- soundfile
- tqdm
- jupyter

See `requirements.txt` for exact versions.

## Documentation

See `CLAUDE.md` for complete technical documentation including:
- Detailed feature extraction algorithms
- Chord parsing and transposition logic
- CNN model architecture
- Training procedures
- Full code examples

## Notes

- **Grouped splitting:** The notebook ensures augmented versions of the same bar stay together in train/val splits
- **Memory efficient:** Processes samples one at a time, suitable for large datasets
- **Progress bars:** Uses tqdm for clear progress indication
- **Flexible:** Easy to modify augmentation settings, feature types, etc.

## Original Project

This is a refactored version of the Deep Learning chord recognition project. See the `Deep Learning/` directory for the original implementation with its complete pipeline.

## License

Educational project for deep learning coursework.
