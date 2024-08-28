import gzip
import os

from rdkit import Chem

from rdkit.Chem import MolStandardize
from rdkit.Chem.SaltRemover import SaltRemover


def curate_compounds(fname):
    basename, _ = os.path.splitext(os.path.splitext(fname)[0])
    curated_output_fname = f"{basename}.smi"
    sremover = SaltRemover()
    lfc = MolStandardize.fragment.LargestFragmentChooser()

    gzsuppl = Chem.ForwardSDMolSupplier(gzip.open(fname))
    smis = []
    for m in gzsuppl:
        if m is None:
            continue
        try:
            mol_ = sremover.StripMol(m, dontRemoveEverything=True)
            lf = lfc.choose(mol_)
            smi = Chem.MolToSmiles(lf)
            cid = m.GetProp('chembl_id').replace("CHEMBL", "")
            smis.append(f"{smi}\t{cid}")
        except:
            continue

    with open(curated_output_fname, 'w') as f:
        f.writelines("\n".join(smis))


def main():
    filename = "./data/chembl_20.sdf.gz"
    curate_compounds(filename)
    print('FINISHED!')


if __name__ == "__main__":
    main()
