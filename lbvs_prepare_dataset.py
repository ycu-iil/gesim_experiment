import glob
import gzip
import os
import re

import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold



def get_smiles_list(filename):
    ret_list = []
    for line in gzip.open(filename, 'rb'):
        line = line.decode('utf-8')
        if line[0] == '#': 
            continue
        # structure of line: [external ID, internal ID, SMILES]]
        ext_id, inter_id, smiles = line.rstrip().split()
        ret_list.append(f"{smiles} {ext_id}")
    return ret_list


def prep_chembl():
    active_files = glob.glob("./data/benchmarking_platform/compounds/ChEMBL/*_actives.dat.gz")
    decoy_file = "./data/benchmarking_platform/compounds/ChEMBL/cmp_list_ChEMBL_zinc_decoys.dat.gz"
    decoy_smiles_list = get_smiles_list(decoy_file)

    pattern = r'ChEMBL_\d+'

    for fname in active_files:
        target_name = re.search(pattern, fname).group()
        base_dir = os.path.dirname(fname)
        output_dir = os.path.join(base_dir, target_name)
        os.makedirs(output_dir, exist_ok=True)
        active_smiles_list = get_smiles_list(fname)
        with open(os.path.join(output_dir, "actives.smi"), 'w') as f:
            f.write("\n".join(active_smiles_list))
        with open(os.path.join(output_dir, "inactives.smi"), 'w') as f:
            f.write("\n".join(decoy_smiles_list))


def prep_dud():
    active_files = glob.glob("./data/benchmarking_platform/compounds/DUD/*_actives.dat.gz")

    pattern = r'DUD_([a-z0-9_]+)_actives'

    for fname in active_files:
        target_name = re.search(pattern, fname).group(1)
        base_dir = os.path.dirname(fname)
        output_dir = os.path.join(base_dir, target_name)
        os.makedirs(output_dir, exist_ok=True)
        active_smiles_list = get_smiles_list(fname)
        decoy_fname = os.path.join(base_dir, f"cmp_list_DUD_{target_name}_decoys.dat.gz")
        if not os.path.exists(decoy_fname):
            raise
        decoy_smiles_list = get_smiles_list(decoy_fname)
        with open(os.path.join(output_dir, "actives.smi"), 'w') as f:
            f.write("\n".join(active_smiles_list))
        with open(os.path.join(output_dir, "inactives.smi"), 'w') as f:
            f.write("\n".join(decoy_smiles_list))


def prep_muv():
    active_files = glob.glob("./data/benchmarking_platform/compounds/MUV/*_actives.dat.gz")

    pattern = r'MUV_\d+'

    for fname in active_files:
        target_name = re.search(pattern, fname).group()
        base_dir = os.path.dirname(fname)
        output_dir = os.path.join(base_dir, target_name)
        os.makedirs(output_dir, exist_ok=True)
        active_smiles_list = get_smiles_list(fname)
        decoy_fname = os.path.join(base_dir, f"cmp_list_{target_name}_decoys.dat.gz")
        if not os.path.exists(decoy_fname):
            raise
        decoy_smiles_list = get_smiles_list(decoy_fname)
        with open(os.path.join(output_dir, "actives.smi"), 'w') as f:
            f.write("\n".join(active_smiles_list))
        with open(os.path.join(output_dir, "inactives.smi"), 'w') as f:
            f.write("\n".join(decoy_smiles_list))


def calc_bms_active_ratio():
    all_active_fnames = glob.glob("./data/benchmarking_platform/compounds/*/*/actives.smi")
    ba_ratio_dict = {}
    for fname in all_active_fnames:
        target_name = os.path.basename(os.path.dirname(fname))
        with open(fname, 'r') as f:
            smiles_list = [l.split(' ')[0] for l in f.readlines()]
            bms_set = set([MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s) for s in smiles_list])
        num_actives = len(smiles_list)
        num_bms = len(bms_set)
        ba_ratio = round(num_bms/num_actives, 2)
        ba_ratio_dict[target_name] = [num_actives, num_bms, ba_ratio]

    df = pd.DataFrame.from_dict(ba_ratio_dict, orient='index', columns=["Num Actives", "Num BMS", "B/A ratio"])
    df.index.name = "Target"
    df.to_csv("./data/benchmarking_platform/compounds/bms_active_ratio.csv")
    

def main():
    print("Dataset Preparation Started...")

    print("ChEMBL...")
    prep_chembl()

    print("DUD...")
    prep_dud()

    print("MUV...")
    prep_muv()

    print("BMS/Active ratio calculation...")
    calc_bms_active_ratio()

    print("DONE!")


if __name__ == "__main__":
    main()
