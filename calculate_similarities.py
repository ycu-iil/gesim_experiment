import glob
import multiprocessing as mp
import os

from gesim import gesim
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions


NUM_CPUS = 130  # Change this variable according to computer specs


def calc_bulk_tanimoto_sim(query_fp, target_fps, output_dir, query_id):
    sim_results = DataStructs.BulkTanimotoSimilarity(query_fp, target_fps)
    out_list = [f"{i}\t{v}\n" for i, v in enumerate(sim_results)]
    ofname = os.path.join(output_dir, f"tanimoto_sims_active{query_id}.txt")
    with open(ofname, 'w') as f:
        f.writelines(out_list)


def calc_bulk_gesim(query_mol, target_mols, output_dir, query_id):
    sim_results = gesim.graph_entropy_similarity_batch(query_mol, target_mols, r=4)
    out_list = [f"{i}\t{v}\n" for i, v in enumerate(sim_results)]
    ofname = os.path.join(output_dir, f"gesims_r4_active{query_id}.txt")
    with open(ofname, 'w') as f:
        f.writelines(out_list)


def main():
    target_base_dir = "./data/benchmarking_platform/compounds/"
    result_base_dir = "./result/benchmarking_platform/"
    dataset_types = ["ChEMBL", "MUV", "DUD"]

    for dtype in dataset_types:
        target_root_dir = os.path.join(target_base_dir, dtype)
        result_root_dir = os.path.join(result_base_dir, dtype)

        target_names = []
        for f in sorted(glob.glob(f"{target_root_dir}/*")):
            if os.path.isdir(f):
                target_names.append(os.path.basename(f))

        for target_name in target_names:
            #if target_name != "ESR1_ant":
            #    continue
            key_amol_dict = {}
            key_afp_dict = {}
            active_mols = []
            active_fps = []
            inactive_mols = []
            inactive_fps = []
            all_mols = None
            all_fps = None

            print(f"Target: {target_name} Process...")
            with open(f"{target_root_dir}/{target_name}/actives.smi", 'r') as f:
                lines = [l.split(' ')[0] for l in f.readlines()]
                #active_mols = [Chem.MolFromSmiles(s) for s in lines[:20]]
                active_mols = [Chem.MolFromSmiles(s) for s in lines]
                print(f"  #actives ({target_name}): {len(active_mols)}")
                key_amol_dict = {k: m for k, m in enumerate(active_mols)}
            with open(f"{target_root_dir}/{target_name}/inactives.smi", 'r') as f:
                lines = [l.split(' ')[0] for l in f.readlines()]
                #inactive_mols = [Chem.MolFromSmiles(s) for s in lines[:100]]
                inactive_mols = [Chem.MolFromSmiles(s) for s in lines]
                print(f"  #inactives ({target_name}): {len(inactive_mols)}")
            all_mols = active_mols + inactive_mols
            num_cpus = len(active_mols) if len(active_mols) < NUM_CPUS else NUM_CPUS

            print("  [PROCESS] Morgan Fingerprint (r=2, dim=2048)...")
            output_dir = f"{result_root_dir}/morgan_fp/result_{target_name}/"
            os.makedirs(output_dir, exist_ok=True)
            active_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in active_mols]
            inactive_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in inactive_mols]
            all_fps = active_fps + inactive_fps
            key_afp_dict = {k: fp for k, fp in enumerate(active_fps)}
            pool = mp.Pool(num_cpus)
            for query_id, query_afp in key_afp_dict.items():
                pool.apply_async(calc_bulk_tanimoto_sim, args=(query_afp, all_fps, output_dir, query_id))
            pool.close()
            pool.join()

            print("  [PROCESS] Feature-connectivity Fingerprint (r=2, dim=2048)...")
            output_dir = f"{result_root_dir}/fc_fp/result_{target_name}/"
            os.makedirs(output_dir, exist_ok=True)
            active_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048, useFeatures=True) for m in active_mols]
            inactive_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048, useFeatures=True) for m in inactive_mols]
            all_fps = active_fps + inactive_fps
            key_afp_dict = {k: fp for k, fp in enumerate(active_fps)}
            pool = mp.Pool(num_cpus)
            for query_id, query_afp in key_afp_dict.items():
                pool.apply_async(calc_bulk_tanimoto_sim, args=(query_afp, all_fps, output_dir, query_id))
            pool.close()
            pool.join()

            print("  [PROCESS] MACCS keys...")
            output_dir = f"{result_root_dir}/maccs_key/result_{target_name}/"
            os.makedirs(output_dir, exist_ok=True)
            active_fps = [MACCSkeys.GenMACCSKeys(m) for m in active_mols]
            inactive_fps = [MACCSkeys.GenMACCSKeys(m) for m in inactive_mols]
            all_fps = active_fps + inactive_fps
            key_afp_dict = {k: fp for k, fp in enumerate(active_fps)}
            pool = mp.Pool(num_cpus)
            for query_id, query_afp in key_afp_dict.items():
                pool.apply_async(calc_bulk_tanimoto_sim, args=(query_afp, all_fps, output_dir, query_id))
            pool.close()
            pool.join()

            print("  [PROCESS] Atom Pair Fingerprint...")
            output_dir = f"{result_root_dir}/atom_pair_fp/result_{target_name}/"
            os.makedirs(output_dir, exist_ok=True)
            active_fps = [Pairs.GetAtomPairFingerprint(m) for m in active_mols]
            inactive_fps = [Pairs.GetAtomPairFingerprint(m) for m in inactive_mols]
            all_fps = active_fps + inactive_fps
            key_afp_dict = {k: fp for k, fp in enumerate(active_fps)}
            pool = mp.Pool(num_cpus)
            for query_id, query_afp in key_afp_dict.items():
                pool.apply_async(calc_bulk_tanimoto_sim, args=(query_afp, all_fps, output_dir, query_id))
            pool.close()
            pool.join()

            print("  [PROCESS] Topological Torsion Fingerprint...")
            output_dir = f"{result_root_dir}/tt_fp/result_{target_name}/"
            os.makedirs(output_dir, exist_ok=True)
            active_fps = [Torsions.GetTopologicalTorsionFingerprintAsIntVect(m) for m in active_mols]
            inactive_fps = [Torsions.GetTopologicalTorsionFingerprintAsIntVect(m) for m in inactive_mols]
            all_fps = active_fps + inactive_fps
            key_afp_dict = {k: fp for k, fp in enumerate(active_fps)}
            pool = mp.Pool(num_cpus)
            for query_id, query_afp in key_afp_dict.items():
                pool.apply_async(calc_bulk_tanimoto_sim, args=(query_afp, all_fps, output_dir, query_id))
            pool.close()
            pool.join()

            print("  [PROCESS] GESim...")
            output_dir = f"{result_root_dir}/ge_sim/result_{target_name}/"
            os.makedirs(output_dir, exist_ok=True)
            pool = mp.Pool(num_cpus)
            for query_id, query_amol in key_amol_dict.items():
                pool.apply_async(calc_bulk_gesim, args=(query_amol, all_mols, output_dir, query_id))
            pool.close()
            pool.join()
    print("DONE!")


if __name__ == "__main__":
    main()
