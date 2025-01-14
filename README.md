# gesim_experiment


## How to setup the Python environment

```bash
git clone git@github.com:ycu-iil/gesim_experiment.git
cd gesim_experiment
conda create -n gesim_expt -c conda-forge python=3.11
# Switch Python virtual environment to gesim_expt
pip install poetry==1.7.1
poetry install --no-root
git clone git@github.com:LazyShion/GESim.git
cd GESim
pip install --upgrade .
```

## How to reproduce experiments

```bash
cd gesim_experiment

# Structural similarity benchmark
# ref: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0148-0
git -C data/ clone git@github.com:nextmovesoftware/similaritybenchmark.git
7zz -odata/similaritybenchmark/ x data/similaritybenchmark/SingleAssay.7z
7zz -odata/similaritybenchmark/ x data/similaritybenchmark/MultiAssay.7z
wget -P ./data https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_20/chembl_20.sdf.gz
python ss_sdf2smi.py
python ss_calculate_single_assay.py
python ss_calculate_multi_assay.py
# Open Jupyter Notebook `ss_analysis.ipynb` and run it.

# LBVS benchmark
# ref: https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-5-26
git -C data/ clone git@github.com:rdkit/benchmarking_platform.git
python lbvs_prepare_dataset.py
python lbvs_calculate_similarities.py
python lbvs_calculate_scores.py
# Open Jupyter Notebook `lbvs_analysis.ipynb` and run it.

# Computation time benchmark
# Open Jupyter Notebook `compare_calculation_time.ipynb` and run it.
```

## Package requirements

- Python: 3.11
- GESim:
- RDKit: 2023.9.1
- Polars: 0.19.15
- pandas: 2.2.0
- matplotlib: 3.8.2
- seaborn: 0.13.2
- jupyter: 1.0.0
- networkx: 3.2.1
