import glob
import multiprocessing as mp
import os

from gesim import gesim
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


NUM_CPUS = 130  # Change this variable according to computer specs
#TARGET_ROOT_DIR = "./data/LIT-PCBA/"
#RESULT_ROOT_DIR = "./result_test/LIT-PCBA/"
#TARGET_ROOT_DIR = "./data/benchmarking_platform/compounds/ChEMBL"
#RESULT_ROOT_DIR = "./result/benchmarking_platform/ChEMBL "
#TARGET_ROOT_DIR = "./data/benchmarking_platform/compounds/DUD"
#RESULT_ROOT_DIR = "./result/benchmarking_platform/DUD"
TARGET_ROOT_DIR = "./data/benchmarking_platform/compounds/MUV"
RESULT_ROOT_DIR = "./result/benchmarking_platform/MUV"


def calc_bulk_tanimoto_sim(query_fp, target_fps, output_dir, query_id):
    sim_results = DataStructs.BulkTanimotoSimilarity(query_fp, target_fps)
    out_list = [f"{i}\t{v}\n" for i, v in enumerate(sim_results)]
    ofname = os.path.join(output_dir, f"tanimoto_sims_active{query_id}.txt")
    with open(ofname, 'w') as f:
        f.writelines(out_list)


def calc_bulk_gesim(query_mol, target_mols, output_dir, query_id, r):
    sim_results = gesim.graph_entropy_similarity_batch(query_mol, target_mols, r=r)
    out_list = [f"{i}\t{v}\n" for i, v in enumerate(sim_results)]
    ofname = os.path.join(output_dir, f"gesims_r{r}_active{query_id}.txt")
    with open(ofname, 'w') as f:
        f.writelines(out_list)


def main():
    target_names = []
    for f in sorted(glob.glob(f"{TARGET_ROOT_DIR}/*")):
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
        with open(f"{TARGET_ROOT_DIR}/{target_name}/actives.smi", 'r') as f:
            lines = [l.split(' ')[0] for l in f.readlines()]
            #active_mols = [Chem.MolFromSmiles(s) for s in lines[:20]]
            active_mols = [Chem.MolFromSmiles(s) for s in lines]
            print(f"  #actives ({target_name}): {len(active_mols)}")
            print("    Morgan fingerprint calculation...")
            active_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in active_mols]
            print("    Done")
            key_amol_dict = {k: m for k, m in enumerate(active_mols)}
            key_afp_dict = {k: fp for k, fp in enumerate(active_fps)}
        with open(f"{TARGET_ROOT_DIR}/{target_name}/inactives.smi", 'r') as f:
            lines = [l.split(' ')[0] for l in f.readlines()]
            #inactive_mols = [Chem.MolFromSmiles(s) for s in lines[:100]]
            inactive_mols = [Chem.MolFromSmiles(s) for s in lines]
            print(f"  #inactives ({target_name}): {len(inactive_mols)}")
            print("    Morgan fingerprint calculation...")
            inactive_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in inactive_mols]
            print("    Done")
        all_mols = active_mols + inactive_mols
        all_fps = active_fps + inactive_fps

        print("  Tanimoto similarity calculation...")
        output_tanimoto_dir = f"{RESULT_ROOT_DIR}/tanimoto_sim/result_{target_name}/"
        os.makedirs(output_tanimoto_dir, exist_ok=True)
        num_cpus = len(active_fps) if len(active_fps) < NUM_CPUS else NUM_CPUS
        pool = mp.Pool(num_cpus)
        for query_id, query_afp in key_afp_dict.items():
            pool.apply_async(calc_bulk_tanimoto_sim, args=(query_afp, all_fps, output_tanimoto_dir, query_id))
        pool.close()
        pool.join()
        print("  Done")

        print("  Graph entropy similarity calculation...")
        output_gesim_dir = f"{RESULT_ROOT_DIR}/ge_sim/result_{target_name}/"
        os.makedirs(output_gesim_dir, exist_ok=True)
        num_cpus = len(active_mols) if len(active_mols) < NUM_CPUS else NUM_CPUS
        radius_list = [1, 2, 3, 4]
        for r in radius_list:
            pool = mp.Pool(num_cpus)
            for query_id, query_amol in key_amol_dict.items():
                pool.apply_async(calc_bulk_gesim, args=(query_amol, all_mols, output_gesim_dir, query_id, r))
            pool.close()
            pool.join()
        print("  Done")


if __name__ == "__main__":
    main()

