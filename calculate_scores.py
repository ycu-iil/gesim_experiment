import glob
import os
import pickle
import random
random.seed(42)
from typing import List, Dict

import numpy as np
import polars as pl
from rdkit.ML.Scoring import Scoring
from rdkit.Chem.Scaffolds import MurckoScaffold


def _merge_files(dirname: str):
    def _extract_number(fname):
        base, _ = os.path.splitext(fname)
        return int(base.split("_active")[-1])
    files = glob.glob(os.path.join(dirname, f"*active*.txt"))
    files_sorted = sorted(files, key=_extract_number)

    with open(files_sorted[0], 'r') as f:
        num_lines = len(f.readlines())
    merged_results = [[str(i)] for i in range(num_lines)]  # initialization with index

    for fname in files_sorted:
        with open(fname, 'r') as f:
            lines = f.readlines()
        for i, l in enumerate(lines):
            _, value = l.split('\t')
            merged_results[i].append(value.strip('\n'))
    ofname = os.path.join(dirname, f"merged_results.txt")
    #print(f"[INFO] Merge results to: {ofname}")
    with open(ofname, 'w') as f:
        f.write('\n'.join('\t'.join(r) for r in merged_results))


def _create_dataframe_from_files(dirname: str) -> pl.DataFrame:
    num_columns = len(glob.glob(os.path.join(dirname, f"*active*.txt")))
    fname = os.path.join(dirname, "merged_results.txt")
    df = pl.read_csv(
        fname,
        has_header=False,
        separator="\t",
        new_columns=['index'] + [str(i) for i in range(num_columns)],
        dtypes=[pl.Int32] + [pl.Float32 for _ in range(num_columns)])
    column_names = df.columns
    column_names.remove('index')
    df = df.with_columns((pl.col('index').is_in([int(i) for i in column_names])).alias('is_active'))
    
    return df


def _calculate_ef_bedroc_auc(
    dirname: str,
    trial_num: int =50,
    sample_num: int =5,
    fractions: List[float] = [0.05],
    alpha: int =20) -> Dict[str, List[float]]:

    ef_list = []
    bedroc_list = []
    auc_list = []

    df = _create_dataframe_from_files(dirname)
    active_column_names = [c for c in df.columns if c not in {'index', 'is_active'}]
    for _ in range(trial_num):
        sampled_columns = random.sample(active_column_names, sample_num)
        bool_list = (df
            .select(
                pl.col('index'), 
                pl.max_horizontal(sampled_columns),
                pl.col('is_active'))
            .sort(
                'max', 
                descending=True,)
            .filter(
                ~pl.col('index').is_in([int(i) for i in sampled_columns]))  # remove training compounds
            .select(
                pl.col('is_active'))
            .to_numpy().tolist()
        )
        
        ef_list.append(Scoring.CalcEnrichment(bool_list, 0, fractions=fractions)[0])
        bedroc_list.append(Scoring.CalcBEDROC(bool_list, 0, alpha=alpha))
        auc_list.append(Scoring.CalcAUC(bool_list, 0))
        
    ret_dict = {
        'enrichment_factor': ef_list,
        'BEDROC': bedroc_list,
        'AUC': auc_list,
    }
    return ret_dict


def main():
    trial_num = 50
    sample_num = 5
    dataset_type = ["ChEMBL", "MUV", "DUD"]
    alpha_list = [20, 100]
    fractions_list = [[0.05], [0.01]]

    for dtype in dataset_type:
        base_dir_list = [d for d in glob.glob(os.path.join("./result/benchmarking_platform/", dtype, "*")) if os.path.isdir(d)]
        
        for base_dir in base_dir_list:
            print(base_dir)
            target_dirs = [d for d in glob.glob(os.path.join(base_dir, "result_*")) if os.path.isdir(d)]
            target_dirs.sort()

            do_merge_files = True
            for alpha, fractions in zip(alpha_list, fractions_list):
                result_dict = {}
                for target_dir in target_dirs:
                    target_name = os.path.basename(target_dir).split("_", 1)[1]
                    print(f"Target name: {target_name} alpha: {alpha} fractions: {fractions}")
                    if do_merge_files:
                        _merge_files(target_dir)
                    result = _calculate_ef_bedroc_auc(
                        target_dir,
                        trial_num=trial_num,
                        sample_num=sample_num,
                        alpha=alpha,
                        fractions=fractions)
                    result_dict[target_name] = result
                do_merge_files = False
                ofname = f"{base_dir}/result_a{alpha}_f{str(fractions[0]).replace('.', '')}.pkl"
                with open(ofname, 'wb') as f:
                    pickle.dump(result_dict, f)
    

if __name__ == "__main__":
    main()
