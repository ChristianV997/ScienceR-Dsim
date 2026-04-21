# ScienceR-Dsim

Integrated topological field simulation + EEG validation system.

## Included
- 3D topology core: Q, Qabs, Q(z,t)
- Defect extraction and worldline tracking
- Reconnection/event detection
- Synthetic validation
- EEG / PCI validation hooks
- Physics hooks (The Well)
- SQLite registry
- Paper figure generation

## Note
Datasets are not embedded. Place them under `data/raw/` using the paths in `data/README.md`.

## Quick start
```bash
pip install -r requirements.txt
python main.py --mode synthetic
python main.py --mode qzt --input data/checkpoints
python main.py --mode eeg --dataset ds002094 --input data/raw/ds002094 --output results/ds002094.csv --compute-pci
python main.py --mode physics --input /path/to/sample.npy --output results/the_well.csv
python main.py --mode cross-domain --results-root results --output results/cross_domain.csv
python paper/generate_figures.py --results-root results --output-dir paper/figures
```

`--mode qzt` writes `results/qzt.csv`, `results/events.csv`, and `results/worldlines.json`.
