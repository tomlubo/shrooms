# Mushroom Classification

Place raw images under `data/merged_dataset/` (ignored by git).
Keep split CSVs in `data/train.csv`, `data/val.csv`, `data/test.csv` (tracked).

## Quickstart
```bash
# create env (conda) and install deps
conda env create -f environment.yml   # or: pip install -r requirements.txt

# train
python train.py --config config.yaml  # if you use a config