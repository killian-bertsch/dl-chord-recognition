# Output Data Directory

This directory will contain the extracted feature CSV files after running the preprocessing notebook.

## Generated Files

After running `data_preprocessing.ipynb`, you will find:

```
data_output/
├── chroma_features.csv     # 12-bin chromagram features
├── hybrid_features.csv     # 36-bin hybrid (LCQT + Chroma) features
├── cqt60_features.csv      # 60-bin CQT features
└── cqt84_features.csv      # 84-bin CQT features
```

## CSV Format

Each CSV file has the following structure:

### Metadata Columns
- `id` - Unique sample identifier (e.g., "beatrice_bar01_shift+2")
- `label` - Full chord label in "root:quality" format (e.g., "F:maj")
- `root` - Root note only (one of: C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B)
- `quality` - Chord quality (maj, min, dom, dim, hdim, aug, sus)
- `song` - Song name
- `shift` - Pitch shift amount in semitones (-2, 0, 2, or 4)

### Feature Columns
- `feature_0` to `feature_N` - Flattened feature array
  - Chroma: 1,044 features (12 bins × 87 frames)
  - Hybrid: 3,132 features (36 bins × 87 frames)
  - CQT60: 5,220 features (60 bins × 87 frames)
  - CQT84: 7,308 features (84 bins × 87 frames)

## Example Row

```csv
id,label,root,quality,song,shift,feature_0,feature_1,feature_2,...
beatrice_bar01,F:maj,F,maj,beatrice,0,-12.5,-15.3,-18.2,...
beatrice_bar01_shift+2,G:maj,G,maj,beatrice,2,-13.1,-14.8,-17.9,...
```

## File Sizes (Approximate)

For a typical dataset with ~500 augmented samples:

- `chroma_features.csv`: ~50 MB
- `hybrid_features.csv`: ~150 MB
- `cqt60_features.csv`: ~250 MB
- `cqt84_features.csv`: ~350 MB

## Usage in Kaggle

1. Upload these CSV files to Kaggle as a dataset
2. In your Kaggle notebook, load them with:

```python
import pandas as pd

# Load features
df_chroma = pd.read_csv('../input/your-dataset/chroma_features.csv')
df_hybrid = pd.read_csv('../input/your-dataset/hybrid_features.csv')

# Extract features and labels
X = df_chroma.filter(regex='feature_').values
y_root = df_chroma['root'].values
y_full = df_chroma['label'].values

# Reshape for CNN: (samples, 1, n_bins, 87)
n_bins = 12  # for chroma
X = X.reshape(-1, 1, n_bins, 87)
```

## Important: Grouped Splitting

When creating train/val splits, use grouped splitting to prevent data leakage:

```python
from sklearn.model_selection import GroupShuffleSplit

# Group by base name (without augmentation suffix)
groups = df_chroma['id'].str.replace(r'_shift.*', '', regex=True)

# Split
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X, y_root, groups))
```

This ensures that all augmented versions of the same bar stay together in either train or val, preventing the model from "cheating."

## Notes

- Features are pre-normalized to 87 time frames
- All values are in dB scale
- Missing/empty directories will result in no output files
