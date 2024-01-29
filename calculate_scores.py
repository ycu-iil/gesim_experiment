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


def _merge_files(dirname: str, useGraphSim: bool =False, radiusGraphSim: int =2):
    def _extract_number(fname):
        base, _ = os.path.splitext(fname)
        return int(base.split("_active")[-1])

    if useGraphSim:
        files = glob.glob(os.path.join(dirname, f"gesims_r{radiusGraphSim}_active*.txt"))
    else:  # tanimoto
        files = glob.glob(os.path.join(dirname, f"tanimoto_sims_active*.txt"))
    files_sorted = sorted(files, key=_extract_number)
    print(files_sorted)

    with open(files_sorted[0], 'r') as f:
        num_lines = len(f.readlines())
    merged_results = [[str(i)] for i in range(num_lines)]  # initialization with index

    print("[INFO] Collect results...")
    for fname in files_sorted:
        with open(fname, 'r') as f:
            lines = f.readlines()
        for i, l in enumerate(lines):
            _, value = l.split('\t')
            merged_results[i].append(value.strip('\n'))
    if useGraphSim:
        ofname = os.path.join(dirname, f"merged_results_r{radiusGraphSim}.txt")
    else:
        ofname = os.path.join(dirname, f"merged_results.txt")
    print(f"[INFO] Dump file: {ofname}")
    with open(ofname, 'w') as f:
        f.write('\n'.join('\t'.join(r) for r in merged_results))


def _create_dataframe_from_files(dirname: str, useGraphSim: bool =False, radiusGraphSim: int =2) -> pl.DataFrame:
    if useGraphSim:
        num_columns = len(glob.glob(os.path.join(dirname, f"gesims_r{radiusGraphSim}_active*.txt")))
        fname = os.path.join(dirname, f"merged_results_r{radiusGraphSim}.txt")
    else:  # tanimoto
        num_columns = len(glob.glob(os.path.join(dirname, f"tanimoto_sims_active*.txt")))
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
    sample_num: int =10,
    fractions: List[float] = [0.05],
    alpha: int =20,
    useGraphSim: bool =False,
    radiusGraphSim: int =2) -> Dict[str, List[float]]:

    ef_list = []
    bedroc_list = []
    auc_list = []

    df = _create_dataframe_from_files(
        dirname,
        useGraphSim=useGraphSim,
        radiusGraphSim=radiusGraphSim)
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

def _calc_scaffoldEF(
    df: pl.DataFrame,
    scaffolds: np.array(str),
    fraction: float =0.05,
    sample_num: int =10,
    trial_num: int =50,
    useGraphSim: bool=False)->float:
    
    scaffold_ef_list = []

    total_num = scaffolds.shape[0]
    total_scaffolds_num = np.unique(scaffolds).shape[0]
    active_column_names = [c for c in df.columns if c not in {'index', 'is_active'}]
    for _ in range(trial_num):
        sampled_columns = random.sample(active_column_names, sample_num)
        top_n_percent_idxs = df.select(
            pl.col('index'),
            pl.min(sampled_columns).apply(lambda x: 1 - x).alias('max') if useGraphSim else pl.max(sampled_columns),
         ).sort(
             'max', descending=True
         ).filter(
             ~pl.col('index').is_in([int(i) for i in sampled_columns])
         ).select(
             pl.col('index')
         ).limit(
             int(len(df) * fraction)
         ).to_pandas().values.flatten()
        subset_num = top_n_percent_idxs.shape[0]
        subset_scaffolds_num = np.unique(scaffolds[top_n_percent_idxs]).shape[0]
        scaffold_ef_list.append((subset_scaffolds_num / subset_num) / (total_scaffolds_num / total_num))
    
    ret_dict = {
        "scaffold_enrichment_factor": scaffold_ef_list
    }
    return ret_dict


def get_scaffolds_from_target_name(dataset_dir: str) -> np.array(str):
    print(f"Target: {dataset_dir} Process...")
    with open(os.path.join(dataset_dir, "actives.smi"), 'r') as f:
        lines = [l.split(' ')[0] for l in f.readlines()]
        actives = [MurckoScaffold.MurckoScaffoldSmiles(s) for s in lines]
    with open(os.path.join(dataset_dir, "inactives.smi"), 'r') as f:
        lines = [l.split(' ')[0] for l in f.readlines()]
        inactives = [MurckoScaffold.MurckoScaffoldSmiles(s) for s in lines]
    return np.array(actives + inactives)


def main():
    radius_list = [2, 3, 4]
    base_dir_list = ["./230623_graphsim/GESim/", "./Tanimoto_RDKit_Results/"]
    alpha_list = [20, 100]
    fractions_list = [[0.05], [0.01]]

    for base_dir in base_dir_list:
        print(base_dir)
        target_dirs = glob.glob(os.path.join(base_dir, "Results_*"))
        target_dirs.sort()

        if base_dir == "./230623_graphsim/GESim/":
            for alpha, fractions in zip(alpha_list, fractions_list):
                for r in radius_list:
                    result_dict = {}
                    for target_dir in target_dirs:
                        target_name = os.path.basename(target_dir).split("_", 1)[1]
                        print(target_name)
                        result = _calculate_ef_bedroc_auc(target_dir, trial_num=50, sample_num=10, alpha=alpha, fractions=fractions,
                                                         useGraphSim=True, radiusGraphSim=r)
                        result_dict[target_name] = result
                    ofname = f"{base_dir}/result_r{r}_a{alpha}_f{str(fractions[0]).replace('.', '')}.pkl"
                    with open(ofname, 'wb') as f:
                        pickle.dump(result_dict, f)
        else:  # "./Tanimoto_RDKit_Results/"
            for alpha, fractions in zip(alpha_list, fractions_list):
                result_dict = {}
                for target_dir in target_dirs:
                    target_name = os.path.basename(target_dir).split("_", 1)[1]
                    print(target_name)
                    result = _calculate_ef_bedroc_auc(target_dir, trial_num=50, sample_num=10, alpha=alpha, fractions=fractions,
                                                     useGraphSim=False, radiusGraphSim=None)
                    result_dict[target_name] = result
                ofname = f"{base_dir}/result_a{alpha}_f{str(fractions[0]).replace('.', '')}.pkl"
                with open(ofname, 'wb') as f:
                    pickle.dump(result_dict, f)
    print("DONE!")

    
def preprocess():
    radius_list = [2, 3, 4]
    base_dir_list = ["./230623_graphsim/GESim/", "./Tanimoto_RDKit_Results/"]

    for base_dir in base_dir_list:
        print(base_dir)
        target_dirs = glob.glob(os.path.join(base_dir, "Results_*"))
        target_dirs.sort()

        if base_dir == "./230623_graphsim/GESim/":
            for r in radius_list:
                for target_dir in target_dirs:
                    target_name = os.path.basename(target_dir).split("_", 1)[1]
                    print(target_name)
                    _merge_files(target_dir, useGraphSim=True, radiusGraphSim=r)
        else:  # "./Tanimoto_RDKit_Results/"
            for target_dir in target_dirs:
                target_name = os.path.basename(target_dir).split("_", 1)[1]
                print(target_name)
                _merge_files(target_dir, useGraphSim=False, radiusGraphSim=None)
    print("DONE!")


def scaffold_calculation():
    #radius_list = [2, 3, 4]
    radius_list = []
    base_dir_list = ["./230623_graphsim/GESim/", "./Tanimoto_RDKit_Results/"]
    fraction = 0.05

    for base_dir in base_dir_list:
        print(base_dir)
        target_dirs = glob.glob(os.path.join(base_dir, "Results_*"))
        target_dirs.sort()

        if base_dir == "./230623_graphsim/GESim/":
            for r in radius_list:
                result_dict = {}
                for target_dir in target_dirs:
                    target_name = os.path.basename(target_dir).split("_", 1)[1]
                    print(target_name)
                    scaffolds = get_scaffolds_from_target_name(target_name)
                    df = _create_dataframe_from_files(target_dir, useGraphSim=True, radiusGraphSim=r)
                    result = _calc_scaffoldEF(df, scaffolds, fraction=fraction, sample_num=10, trial_num=50,
                                              useGraphSim=True)
                    result_dict[target_name] = result
                ofname = f"{base_dir}/result_sef_r{r}_f{str(fraction).replace('.', '')}.pkl"
                with open(ofname, 'wb') as f:
                    pickle.dump(result_dict, f)
        else:  # "./Tanimoto_RDKit_Results/"
            result_dict = {}
            for target_dir in target_dirs:
                target_name = os.path.basename(target_dir).split("_", 1)[1]
                print(target_name)
                scaffolds = get_scaffolds_from_target_name(target_name)
                df = _create_dataframe_from_files(target_dir, useGraphSim=False, radiusGraphSim=None)
                result = _calc_scaffoldEF(df, scaffolds, fraction=fraction, sample_num=10, trial_num=50,
                                          useGraphSim=False)
                result_dict[target_name] = result
            ofname = f"{base_dir}/result_sef_f{str(fraction).replace('.', '')}.pkl"
            with open(ofname, 'wb') as f:
                pickle.dump(result_dict, f)
    print("DONE!")


if __name__ == "__main__":
    #main()
    #preprocess()
    scaffold_calculation()
