import requests 
import pandas as pd
import json
from Bio import PDB
# so we need to load in the amino acid sequence from uniprot, and get the two structures from pdb. Then we align them    
def load_pdb(pdb_plus_chain):
    # some pdb ids might be formatted like this
    pdb_plus_chain = pdb_plus_chain.replace(":", "_")
    if("_" not in pdb_plus_chain):
        return ValueError(f"bad id: {pdb_plus_chain}")
    pdb_id, chain_id = pdb_plus_chain.split("_")
    pdb_id = pdb_id[:4]
    pdbl = PDB.PDBList()
    file_path = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=".")
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, file_path) 
    chain = structure[0][chain_id.upper()] 
    atoms = {}
    res_names = {}
    for residue in chain:
        if residue.id[0] != " ":
            continue
        
        res_num = residue.id[1]
        
        if "CA" in residue:
            res_names[res_num] = residue.get_resname()

            atoms[res_num] = residue["CA"]
            
    return atoms, res_names


def load_motion_structures(pdb1, pdb2):
    atoms1, res_names1 = load_pdb(pdb1)
    atoms2, res_names2 = load_pdb(pdb2)
    # TODO Implement this
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    alignment = aligner.align(seq1_str, seq2_str)[0]  
    intersection = set(atoms1.keys()) & set(atoms2.keys())
    #print("diff1: ",[res_names1[key] for key in sorted(list(set(atoms1.keys()) - intersection))])
    #print("diff2: ",[res_names2[key] for key in sorted(list(set(atoms2.keys()) - intersection))])
    common_res_numbers = sorted(list(intersection))
    
    fixed_atoms = [atoms1[num] for num in common_res_numbers]
    moving_atoms = [atoms2[num] for num in common_res_numbers]
    
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(fixed_atoms, moving_atoms)

    super_imposer.apply(moving_atoms) 
    print(f"Alignment Complete. RMSD: {super_imposer.rms:.4f} Å")

     aligned_coords = {
        str(fixed.get_parent().id): {  # Extracts the residue ID tuple from the atom safely
            "fixed_coord": fixed.get_coord().tolist(),
            "moving_coord_aligned": moving.get_coord().tolist()
        } for fixed, moving in zip(fixed_atoms, moving_atoms)
    }
    
    return aligned_coords, 

# for temporary use
if __name__ == "__main__":
    print(load_motion_structures("6WGU_A", "6WH5_A"))
    

    

