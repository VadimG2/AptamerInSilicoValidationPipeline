import sys
import os
import warnings
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.DSSP import DSSP

warnings.simplefilter('ignore', BiopythonWarning)

# Cписки остатков
PROTEIN_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'CYX', 'HID', 'HIE', 'HIP', 'ASH', 'GLH', 'LYN'
}

RNA_RESIDUES = {
    'A', 'G', 'C', 'U', 'I',
    'DA', 'DG', 'DC', 'DT', 'DI',
    'ADE', 'GUA', 'CYT', 'URA', 'THY', 
    'RA', 'RG', 'RC', 'RU'
}

class ProteinSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname().strip().upper() in PROTEIN_RESIDUES

class RNASelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname().strip().upper() in RNA_RESIDUES

def debug_print(msg):
    sys.stderr.write(f"[DEBUG] {msg}\n")

def run_prep(input_pdb, output_prot, output_rna, output_ss):
    debug_print(f"Processing {input_pdb}")
    
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('complex', input_pdb)
    except Exception as e:
        debug_print(f"CRITICAL ERROR parsing PDB: {e}")
        sys.exit(1)

    model = structure[0]
    all_residues = list(model.get_residues())
    if not all_residues:
        debug_print("ERROR: No residues found. Creating empty outputs to prevent pipeline crash.")
        with open(output_prot, 'w') as f: f.write("END\n")
        with open(output_rna, 'w') as f: f.write("END\n")
        with open(output_ss, 'w') as f: f.write("C\n")
        return    

    unique_resnames = set(r.get_resname().strip().upper() for r in all_residues)
    debug_print(f"Found residue types in PDB: {unique_resnames}")
    
    # 1. Переименовываем CYX -> CYS
    for residue in all_residues:
        resname = residue.get_resname().strip().upper()
        if resname == 'CYX':
            residue.resname = 'CYS'
        
        for atom in residue:
            if atom.occupancy == 0.0 or atom.occupancy is None:
                atom.occupancy = 1.0
            if atom.bfactor == 0.0 or atom.bfactor is None:
                atom.bfactor = 25.0

    io = PDBIO()
    io.set_structure(model)

    # 2. Сохраняем белок
    temp_prot = output_prot + ".temp"
    try:
        io.save(temp_prot, select=ProteinSelect())
    except Exception as e:
        debug_print(f"Error saving protein part: {e}")

    has_protein = False
    if os.path.exists(temp_prot) and os.path.getsize(temp_prot) > 0:
        with open(temp_prot, 'r') as f:
            if any(line.startswith("ATOM") for line in f):
                has_protein = True

    if not has_protein:
        debug_print("ERROR: No protein residues found! (Check 'Found residue types' above)")
        debug_print("Expected protein residues: " + str(sorted(list(PROTEIN_RESIDUES))))
        sys.exit(1)

    # Добавляем хедер
    header = 'HEADER    PREDICTED MODEL ESMFOLD/HDOCK      14-NOV-25   PROT   '
    cryst1 = 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f%11s%4d' % (10.0, 10.0, 10.0, 90.0, 90.0, 90.0, 'P 1', 1)
    
    with open(temp_prot, 'r') as f_in, open(output_prot, 'w') as f_out:
        f_out.write(header + "\n")
        f_out.write(cryst1 + "\n")
        for line in f_in:
            if line.startswith("ATOM") or line.startswith("TER"):
                f_out.write(line)
        f_out.write("ENDMDL\n")
    
    if os.path.exists(temp_prot): os.remove(temp_prot)

    # 3. Сохраняем РНК
    try:
        io.save(output_rna, select=RNASelect())
    except Exception as e:
        debug_print(f"Error saving RNA part: {e}")

    has_rna = False
    if os.path.exists(output_rna) and os.path.getsize(output_rna) > 0:
        with open(output_rna, 'r') as f:
             if any(line.startswith("ATOM") for line in f):
                 has_rna = True
    
    if not has_rna:
        debug_print("WARNING: No RNA residues found. Creating dummy file.")
        with open(output_rna, 'w') as f:
            f.write("END\n")
    else:
        with open(output_rna, "a") as f:
            f.write("END\n")

    # 4. DSSP
    debug_print("Running DSSP...")
    try:
        prot_structure = parser.get_structure('prot_only', output_prot)
        model_prot = prot_structure[0]
        
        dssp = DSSP(model_prot, output_prot, dssp='dssp')
        ss_line = ''
        if len(dssp) > 0:
            for key in sorted(dssp.keys()):
                ss_code = dssp[key][2]
                ss_line += ss_code
        else:
            raise ValueError("DSSP returned empty result")

        ss_martini = ss_line.replace('-', 'C').replace('P', 'C')
        
        with open(output_ss, 'w') as f:
            f.write(ss_martini)
        debug_print("DSSP success.")
            
    except Exception as e:
        debug_print(f"Warning: DSSP failed ({e}). using Coil (C).")
        n_res = 0
        try:
             with open(output_prot) as f:
                 for line in f:
                     if line.startswith("ATOM") and "CA" in line: # Грубый подсчет по C-alpha
                         n_res += 1
        except:
            n_res = 1
        if n_res == 0: n_res = 1
        
        with open(output_ss, 'w') as f:
            f.write("C" * n_res)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python prepare_structure.py input.pdb out_prot.pdb out_rna.pdb ss.txt")
        sys.exit(1)
        
    run_prep(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
