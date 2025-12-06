# Chord Recognition Workflow

## Visual Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT DATA SOURCES                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ├─── data_input/00_raw/
                              │    (Ableton exports: slices + chords)
                              │
                              └─── data_input/00_input/
                                   (MusicXML + full audio)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              data_preprocessing.ipynb (ONE NOTEBOOK)            │
│─────────────────────────────────────────────────────────────────│
│                                                                 │
│  STEP 1: Load Raw Data                                         │
│  ├─ Read chord progressions                                    │
│  ├─ Load audio slices                                          │
│  └─ Parse chord symbols → (root, quality)                      │
│                                                                 │
│  STEP 2: Clean & Organize                                      │
│  ├─ Deduplicate Ableton exports                                │
│  ├─ Split bars with 2 chords in half                           │
│  └─ Create sample IDs                                          │
│                                                                 │
│  STEP 3: Augmentation (4x increase)                            │
│  ├─ Pitch shift: -2, 0, +2, +4 semitones                       │
│  └─ Transpose labels accordingly                               │
│                                                                 │
│  STEP 4: Feature Extraction (4 types)                          │
│  ├─ Chroma (12 bins)                                           │
│  ├─ Hybrid (36 bins = 24 LCQT + 12 Chroma)                     │
│  ├─ CQT60 (60 bins)                                            │
│  └─ CQT84 (84 bins)                                            │
│                                                                 │
│  STEP 5: Export to CSV                                         │
│  ├─ Flatten features: (n_bins, 87) → flat array                │
│  ├─ Add metadata: id, label, root, quality, song, shift        │
│  └─ Save one CSV per feature type                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT CSV FILES                              │
│─────────────────────────────────────────────────────────────────│
│  data_output/                                                   │
│  ├─ chroma_features.csv  (12 × 87 = 1,044 features/sample)     │
│  ├─ hybrid_features.csv  (36 × 87 = 3,132 features/sample)     │
│  ├─ cqt60_features.csv   (60 × 87 = 5,220 features/sample)     │
│  └─ cqt84_features.csv   (84 × 87 = 7,308 features/sample)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Upload to Kaggle Datasets
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   KAGGLE TRAINING NOTEBOOK                      │
│─────────────────────────────────────────────────────────────────│
│                                                                 │
│  1. Load CSV                                                    │
│     df = pd.read_csv('../input/your-dataset/chroma.csv')       │
│                                                                 │
│  2. Extract Features & Labels                                  │
│     X = df.filter(regex='feature_').values                     │
│     y = df['root'].values                                      │
│                                                                 │
│  3. Grouped Train/Val Split (prevents data leakage!)           │
│     groups = df['id'].str.replace(r'_shift.*', '')             │
│     splitter = GroupShuffleSplit(...)                          │
│                                                                 │
│  4. Reshape for CNN                                            │
│     X = X.reshape(-1, 1, n_bins, 87)                           │
│                                                                 │
│  5. Build & Train CNN                                          │
│     model = ChordCNN(num_classes=12, input_bins=n_bins)        │
│     model.fit(X_train, y_train)                                │
│                                                                 │
│  6. Evaluate & Save                                            │
│     accuracy = model.evaluate(X_val, y_val)                    │
│     model.save('best_model.pt')                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Details

### Input Format

**Raw Data (00_raw/song_name/):**
```
beatrice/
├── beatrice_chords          ← Chord progression
│   Fmaj7
│   Gbmaj7
│   Dm7 | Cm7                ← Two chords, will be split
│   ...
│
└── beatrice_clips/          ← Audio slices
    ├── Slice 1 [...].wav    ← Bar 1
    ├── Slice 2 [...].wav    ← Bar 2
    ├── Slice 3 [...].wav    ← Bar 3
    └── ...
```

### Processing Transformations

```
Original Sample:
├─ ID: beatrice_bar01
├─ Audio: [1-bar WAV]
└─ Label: F:maj

    ↓ Augmentation (4 pitch shifts)

Augmented Samples:
├─ beatrice_bar01 (shift 0)           → F:maj
├─ beatrice_bar01_shift-2 (shift -2)  → Eb:maj
├─ beatrice_bar01_shift+2 (shift +2)  → G:maj
└─ beatrice_bar01_shift+4 (shift +4)  → A:maj

    ↓ Feature Extraction

Features (for chroma):
├─ Shape: (12, 87)
├─ Normalized to 87 frames
└─ Values in dB scale

    ↓ CSV Export

CSV Row:
id,label,root,quality,song,shift,feature_0,feature_1,...,feature_1043
beatrice_bar01_shift+2,G:maj,G,maj,beatrice,2,-12.5,-15.3,...,-20.1
```

### Output CSV Structure

```csv
id,label,root,quality,song,shift,feature_0,feature_1,...,feature_N
beatrice_bar01,F:maj,F,maj,beatrice,0,-12.5,-15.3,...
beatrice_bar01_shift+2,G:maj,G,maj,beatrice,2,-13.1,-14.8,...
beatrice_bar01_shift+4,A:maj,A,maj,beatrice,4,-11.9,-16.2,...
beatrice_bar02,Gb:maj,Gb,maj,beatrice,0,-14.2,-13.5,...
...
```

## Workflow Steps

### Local Machine (Preprocessing)

1. **Prepare Environment**
   ```bash
   cd Chord_Recognition_Refactored
   pip install -r requirements.txt
   ```

2. **Copy Input Data**
   ```bash
   cp -r "../Deep Learning/data/00_raw/"* data_input/00_raw/
   ```

3. **Run Preprocessing Notebook**
   ```bash
   jupyter notebook data_preprocessing.ipynb
   # Run all cells
   ```

4. **Verify Output**
   ```bash
   ls -lh data_output/
   # Should see 4 CSV files
   ```

### Kaggle (Training)

1. **Upload Dataset**
   - Go to kaggle.com/datasets
   - Upload CSV files from `data_output/`
   - Make dataset public or private

2. **Create Notebook**
   - New Notebook → Add dataset
   - Import necessary libraries

3. **Load & Prepare Data**
   ```python
   # Load
   df = pd.read_csv('../input/chord-features/chroma_features.csv')

   # Split features and labels
   X = df.filter(regex='feature_').values
   y = df['root'].map({r: i for i, r in enumerate(CHROMATIC_SCALE)}).values

   # Reshape
   X = X.reshape(-1, 1, 12, 87)
   ```

4. **Grouped Splitting**
   ```python
   from sklearn.model_selection import GroupShuffleSplit

   groups = df['id'].str.replace(r'_shift.*', '', regex=True)
   splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
   train_idx, val_idx = next(splitter.split(X, y, groups))
   ```

5. **Build & Train CNN**
   ```python
   import torch
   import torch.nn as nn

   # Define model (see CLAUDE.md for architecture)
   model = ChordCNN(num_classes=12, input_bins=12)

   # Train
   trainer = Trainer(model, train_loader, val_loader)
   trainer.fit(epochs=100)
   ```

6. **Evaluate & Save**
   ```python
   # Evaluate
   val_acc = trainer.evaluate(val_loader)
   print(f"Validation Accuracy: {val_acc:.2f}%")

   # Save
   torch.save(model.state_dict(), 'best_model.pt')
   ```

## Timeline

```
┌──────────────┐
│   10 min     │  Setup environment
├──────────────┤
│   2 min      │  Copy input data
├──────────────┤
│  30-60 min   │  Run preprocessing notebook
│              │  (depends on dataset size)
├──────────────┤
│   5 min      │  Upload to Kaggle
├──────────────┤
│  30-120 min  │  Train models on Kaggle
│              │  (depends on feature type & epochs)
└──────────────┘
Total: ~1.5-3 hours for complete pipeline
```

## Key Advantages

### Before (Original)
```
5 data directories
7 Python scripts
Complex dependencies
Local training only
.npy files (not portable)
```

### After (Refactored)
```
2 data directories
1 Jupyter notebook
Simple workflow
Cloud training ready
CSV files (portable)
```

## Common Issues & Solutions

### Issue: No raw data
**Solution:** Copy from original project
```bash
cp -r "../Deep Learning/data/00_raw/"* data_input/00_raw/
```

### Issue: Memory error during preprocessing
**Solution:** Reduce number of songs or process in batches
```python
# In notebook, modify to process subset
song_dirs = song_dirs[:5]  # Process first 5 songs only
```

### Issue: CSV files too large
**Solution:** Use only chroma or hybrid features (smaller)
```python
# Modify FEATURE_TYPES
FEATURE_TYPES = ['chroma', 'hybrid']  # Skip cqt60 and cqt84
```

### Issue: Data leakage in training
**Solution:** Always use grouped splitting
```python
# Group by base name without augmentation suffix
groups = df['id'].str.replace(r'_shift.*', '', regex=True)
```

## Feature Type Comparison

| Feature | Bins | CSV Size | Training Speed | Accuracy |
|---------|------|----------|----------------|----------|
| Chroma  | 12   | ~50 MB   | Fastest        | Good     |
| Hybrid  | 36   | ~150 MB  | Fast           | Best     |
| CQT60   | 60   | ~250 MB  | Medium         | Good     |
| CQT84   | 84   | ~350 MB  | Slower         | Good     |

**Recommendation:** Start with Chroma for quick iteration, use Hybrid for best results.

## Next Steps

After training on Kaggle:
1. Download trained model weights
2. Use for inference on new audio
3. Compare performance across feature types
4. Experiment with different augmentation strategies
5. Try ensemble methods

See `CLAUDE.md` for complete technical documentation!
