# Dataset Folder (Not Uploaded to GitHub)

Put Kaggle dataset CSV files here.

Supported formats:

## Option 1: Fake.csv + True.csv
- `Fake.csv` (Fake=0)
- `True.csv` (Real=1)

## Option 2: train.csv
- `train.csv` with a `label` column where `0=fake` and `1=real`

## Optional: Add latest labeled samples
Create `custom_labeled.csv` with columns:
- `text,label`
- `label`: `0=fake`, `1=real`

Note: Large CSV files are ignored by `.gitignore` and will not be pushed to GitHub.
