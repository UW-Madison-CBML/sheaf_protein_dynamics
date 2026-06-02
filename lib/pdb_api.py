import requests 
import pandas as pd
import json
from Bio import PDB, Align
from Bio.SeqUtils import seq1

#TODO: so we need to load in the amino acid sequence from uniprot, and get the two structures from pdb. Then we align them    
    
def load_pdb(pdb_plus_chain):
    # pdb ids are sometimes formatted like this
    pdb_plus_chain = pdb_plus_chain.replace(":", "_")
    if("_" not in pdb_plus_chain):
        print(f"no chain id: {pdb_plus_chain}")
        pdb_id = pdb_plus_chain
        chain_id = ""
    else:
        pdb_id, chain_id = pdb_plus_chain.split("_")
    pdb_id = pdb_id[:4]
    pdbl = PDB.PDBList()
    file_path = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=".")
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, file_path) 
    if chain_id:
        chain = structure[0][chain_id.upper()] 
    else:
        first_structure = structure[0]
        all_chains = list(first_structure.get_chains())
    
        if all_chains:
            chain_id = all_chains[0].id.strip()
        else:
            chain_id = "A" # otherwise fall back to A, this does not seem to happen much
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

    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1

    seq1_list = [(num, seq1(name)) for num, name in res_names1.items() if seq1(name) != 'X']
    seq2_list = [(num, seq1(name)) for num, name in res_names2.items() if seq1(name) != 'X']
    
    seq1_str = "".join([char for _, char in seq1_list])
    seq2_str = "".join([char for _, char in seq2_list])

    intersecting_atoms1 = []
    intersecting_atoms2 = []

    alignment = aligner.align(seq1_str, seq2_str)[0]  
    blocks1, blocks2 = alignment.aligned[:2]


    for block1, block2 in zip(blocks1, blocks2):
        for idx in range(block1[1] - block1[0]): # block1[1] - block1[0] = block2[1] - block2[0]
            idx1 = idx + block1[0]
            idx2 = idx + block2[0]

            atom1 = atoms1[seq1_list[idx1][0]]
            atom2 = atoms2[seq2_list[idx2][0]]
            if(atom1.get_parent().get_resname() == atom2.get_parent().get_resname()):
                intersecting_atoms1.append(atom1)
                intersecting_atoms2.append(atom2)
            
    
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(intersecting_atoms1, intersecting_atoms2)

    super_imposer.apply(intersecting_atoms2) 

    
    return intersecting_atoms1, intersecting_atoms2, [atom.get_parent().get_resname() for atom in intersecting_atoms1]

# for temporary use
if __name__ == "__main__":
    print(load_motion_structures("2V66", "2V66"))
    

    

