import requests 
import pandas as pd
import numpy as np
import json
from Bio import PDB, Align
from Bio.SeqUtils import seq1
import sys
import time
import os
#TODO: so we need to load in the amino acid sequence from uniprot, and get the two structures from pdb. Then we align them    
    
def retrieve_pdb_file(pdb_id, file_format = "cif"):
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.{file_format}"
    
    # TODO remove in case of anonymization
    headers = {
        "User-Agent": "jlundsgaard@wisc.edu"
    }
    file_path = os.path.abspath(f"{pdb_id}.{file_format}")

    i = 0
    attempts = 8
    delay = 1
    while(i < attempts):
        try:
            if (response := requests.get(url, headers=headers)).status_code != 200:
                time.sleep(delay)
            else:
                with open(file_path, "w") as file:
                    file.write(response.text)
                return file_path
        except requests.exceptions.ReadTimeout:
            time.sleep(delay)
        i += 1
        delay *= 1.3
    raise ValueError(f"{pdb_id} could not be accessed at {url}: error code {response.status_code}")


def load_pdb(pdb_plus_chain):
    # pdb ids are sometimes formatted like this
    pdb_plus_chain = pdb_plus_chain.replace(":", "_")
    if("_" not in pdb_plus_chain):
        print(f"no chain id: {pdb_plus_chain}")
        pdb_id = pdb_plus_chain
        chain_id = ""
    else:
        pdb_id, chain_id = pdb_plus_chain.split("_")
    pdb_id = pdb_id[:4].upper()

    # uncomment if using biopython PDB for api access
    #pdbl = PDB.PDBList()
    #file_path = pdbl.retrieve_pdb_file(pdb_id)

    file_path = retrieve_pdb_file(pdb_id)

    # uncomment if using old .pdb format
    # parser = PDB.PDBParser(QUIET=True)
    parser = PDB.MMCIFParser(QUIET=True) 
    structure = parser.get_structure(pdb_id, file_path) 
    first_structure = structure[0]
    all_chains = list(first_structure.get_chains())

    if chain_id and chain_id in all_chains:
        chain = first_structure[chain_id]  
    else:
        if all_chains:
            chain_id = all_chains[0].id.strip()
        else:
            chain_id = "A" # otherwise fall back to A, this does not seem to happen much
        chain_id = chain_id.upper()
        chain = first_structure[chain_id]
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

def pdb_chain_to_uniprot_id(pdb_plus_chain):

    pdb_plus_chain = pdb_plus_chain.replace(":", "_")

    if "_" not in pdb_plus_chain:
        # Fall back to chain 'A'
        pdb_id = pdb_plus_chain
        target_chain = "A"
    else:
        pdb_id, target_chain = pdb_plus_chain.split("_")
    
    pdb_id = pdb_id.lower()
    target_chain = target_chain.upper()

    # Query SIFTS API
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    response = requests.get(url)


    if response.status_code == 200:
        data = response.json()
        try:
            uniprot_dict = data[pdb_id]['UniProt']
        
            for uniprot_id, mapping_data in uniprot_dict.items():
                for mapping in mapping_data['mappings']:
                    if mapping['chain_id'] == target_chain:
                        return uniprot_id # found exact match
            
            print(f"Chain {target_chain} not found in UniProt mappings for PDB {pdb_id.upper()}.")
            return None
        
        except KeyError:
            print(f"Mapping structure missing for PDB {pdb_id.upper()}.")
            return None

    else:
        print(f"Failed to fetch data for {pdb_id.upper()}. HTTPS Status: {response.status_code}")
        return None  

def get_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)

    if response.status_code == 200:
        # FASTA format
        fasta_lines = response.text.strip().split('\n')
        sequence = ''.join(fasta_lines[1:])
        return sequence
    else:
        print(f"Error fetching data: HTTP {response.status_code}")
        return None
    
def map_pdb_to_uniprot(atoms, res_names, uniprot_seq:str):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1

    # PDB sequence
    seq_list = [(num, seq1(name)) for num, name in res_names.items() if seq1(name) != 'X']

    seq_str = "".join([char for _, char in seq_list])

    # uniprot seq is already string
    alignment = aligner.align(uniprot_seq, seq_str)[0]
    uniprot_blocks, pdb_blocks = alignment.aligned[:2]

    aligned_atoms = [None] * len(uniprot_seq)

    for u_block, p_block in zip(uniprot_blocks, pdb_blocks):
        for i in range(u_block[1] - u_block[0]):
            u_idx = u_block[0] + i
            p_idx = p_block[0] + i

            res_num = seq_list[p_idx][0]
            if res_num in atoms:
                aligned_atoms[u_idx] = atoms[res_num]
    
    return aligned_atoms



def load_motion_structures(pdb1, pdb2, uniprot_seq:str):

    atoms1, res_names1 = load_pdb(pdb1)
    atoms2, res_names2 = load_pdb(pdb2)

    aligned_atoms1 = map_pdb_to_uniprot(atoms1, res_names1, uniprot_seq)
    aligned_atoms2 = map_pdb_to_uniprot(atoms2, res_names2, uniprot_seq)

    intersecting_atoms1 = []
    intersecting_atoms2 = []

    for a1, a2 in zip(aligned_atoms1, aligned_atoms2):
        if a1 is not None and a2 is not None:
            intersecting_atoms1.append(a1)
            intersecting_atoms2.append(a2)
    
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(intersecting_atoms1, intersecting_atoms2)

    # Apply rotation/translation to all valid atoms in conformation 2
    valid_atoms2 = [a for a in aligned_atoms2 if a is not None]
    super_imposer.apply(valid_atoms2) 

    # Need to pad the final coords to line up pdb position with uniprot seq
    L = len(uniprot_seq)
    conformation1 = np.zeros((L,3))
    conformation2 = np.zeros((L,3))

    for i in range(L):
        # Extract conf 1
        if aligned_atoms1[i] is not None:
            conformation1[i] = aligned_atoms1[i].get_coord()
        else:
            conformation1[i] = [-1.0, -1.0, -1.0] # Pad if empty
        
        # Conf 2
        if aligned_atoms2[i] is not None:
            conformation2[i] = aligned_atoms2[i].get_coord()
        else:
            conformation2[i] = [-1.0, -1.0, -1.0]
    
    return conformation1, conformation2 

# for temporary use
if __name__ == "__main__":
    pdb_list = [
    '4cfo',
    '4cfp',
    '6d6f',
    '4tws',
    '5a1r',
    '4ny4',
    '3kjr',
    '3k2h',
    '1fit',
    '6fit',
    '2bjs',
    '1odn',
    '2qyo',
    '4h4t',
    '4h50',
    '5tcg',
    '5tci',
    '4k4r',
    '4k4p',
    '5lcq',
    '5n1n',
    '3v4n',
    '1ysl',
    '4rzu',
    '4rzu']

    for i in pdb_list:
        uniprot_id = pdb_chain_to_uniprot_id(i)
        uniprot_seq = get_uniprot_sequence(uniprot_id)
        print(load_motion_structures(i, i, uniprot_seq))
    
