# gesim_experiment

## Preparation

### Package requirements

- Python: 3.11
- GESim:
- Polars: 0.19.15

### How to reproduce

```bash
git clone git@github.com:ycu-iil/gesim_experiment.git
cd gesim_experiment
git -C data/ clone git@github.com:rdkit/benchmarking_platform.git
python prepare_dataset.py
python calculate_similarities.py
python calculate_scores.py 
```
