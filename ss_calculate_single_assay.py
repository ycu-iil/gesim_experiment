import glob
import pickle
import os
import re

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions
from scipy.stats import spearmanr

from gesim import gesim


def extract_number(f):
    return int(re.search(r'\d+', f).group())


def main():
    print("[INFO] start...")
    files = sorted(glob.glob("./data/similaritybenchmark/SingleAssay/dataset/*.txt"), key=extract_number)
    uniq_num_single_set = set()
    for fname in files:
        with open(fname, 'r') as f:
            lines = [l.strip('\n') for l in f.readlines()]
        for l in lines:
            numbers = l.split()
            uniq_num_single_set.update(map(int, numbers))

    with open('./data/chembl_20.smi', 'r') as f:
        lines = [l.strip('\n').split('\t') for l in f.readlines()]
    used_chembl_ids = uniq_num_single_set
    id_mol_dict = {}
    for smi, k in lines:
        if int(k) in used_chembl_ids:
            id_mol_dict[k] = Chem.MolFromSmiles(smi)

    num_use_files = 1000
    result_dict = {
        "mfp": [[] for _ in range(num_use_files)],
        "fcfp": [[] for _ in range(num_use_files)],
        "maccs": [[] for _ in range(num_use_files)],
        "apfp": [[] for _ in range(num_use_files)],
        "ttfp": [[] for _ in range(num_use_files)],
        "gesim": [[] for _ in range(num_use_files)],
        "gesim_r3": [[] for _ in range(num_use_files)],
    }

    for trial_num, fname in enumerate(files[:num_use_files]):
        with open(fname, 'r') as f:
            lines = [l.strip('\n') for l in f.readlines()]

        used_num_sigle = 4563
        ref_rank = range(4, 0, -1)
        for l in lines[:used_num_sigle]:
            sim_rank_list = l.split()
            sim_mol_rank_list = [id_mol_dict[i] for i in sim_rank_list]

            # Morgan Fingerprint (r=2, dim=2048)
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in sim_mol_rank_list]
            sim_list = DataStructs.BulkTanimotoSimilarity(fps[0], fps[1:])
            if len(set(sim_list)) == 1:
                result_dict['mfp'][trial_num].append(0)
            else: 
                result_dict['mfp'][trial_num].append(spearmanr(ref_rank, sim_list).statistic)

            # Feature-connectivity Fingerprint (r=2, dim=2048)
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048, useFeatures=True) for m in sim_mol_rank_list]
            sim_list = DataStructs.BulkTanimotoSimilarity(fps[0], fps[1:])
            if len(set(sim_list)) == 1:
                result_dict['fcfp'][trial_num].append(0)
            else:
                result_dict['fcfp'][trial_num].append(spearmanr(ref_rank, sim_list).statistic)

            # MACCS keys
            fps = [MACCSkeys.GenMACCSKeys(m) for m in sim_mol_rank_list]
            sim_list = DataStructs.BulkTanimotoSimilarity(fps[0], fps[1:])
            if len(set(sim_list)) == 1:
                result_dict['maccs'][trial_num].append(0)
            else:
                result_dict['maccs'][trial_num].append(spearmanr(ref_rank, sim_list).statistic)

            # Atom Pair Fingerprint
            fps = [Pairs.GetAtomPairFingerprint(m) for m in sim_mol_rank_list]
            sim_list = DataStructs.BulkTanimotoSimilarity(fps[0], fps[1:])
            if len(set(sim_list)) == 1:
                result_dict['apfp'][trial_num].append(0)
            else:
                result_dict['apfp'][trial_num].append(spearmanr(ref_rank, sim_list).statistic)

            # Topological Torsion Fingerprint
            fps = [Torsions.GetTopologicalTorsionFingerprintAsIntVect(m) for m in sim_mol_rank_list]
            sim_list = DataStructs.BulkTanimotoSimilarity(fps[0], fps[1:])
            if len(set(sim_list)) == 1:
                result_dict['ttfp'][trial_num].append(0)
            else:
                result_dict['ttfp'][trial_num].append(spearmanr(ref_rank, sim_list).statistic)

            # GESim
            sim_list = gesim.graph_entropy_similarity_batch(sim_mol_rank_list[0], sim_mol_rank_list[1:], r=4, uncertainty=1)
            if len(set(sim_list)) == 1:
                result_dict['gesim'][trial_num].append(0)
            else:
                result_dict['gesim'][trial_num].append(spearmanr(ref_rank, sim_list).statistic)

            # GESim (r=3)
            sim_list = gesim.graph_entropy_similarity_batch(sim_mol_rank_list[0], sim_mol_rank_list[1:], r=3, uncertainty=1)
            if len(set(sim_list)) == 1:
                result_dict['gesim_r3'][trial_num].append(0)
            else:
                result_dict['gesim_r3'][trial_num].append(spearmanr(ref_rank, sim_list).statistic)

    output_file = 'result/similaritybenchmark/result_single_assay.pkl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"[INFO] Save a result: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(result_dict, f)
    print("[INFO] Finish!")


if __name__ == "__main__":
    main()