import sys
import os
import subprocess

def run_strict_prep(input_pdb, output_prot, output_rna, output_ss):
    input_pdb = os.path.abspath(input_pdb)
    output_prot = os.path.abspath(output_prot)
    output_rna = os.path.abspath(output_rna)
    output_ss = os.path.abspath(output_ss)

    work_dir = os.path.dirname(output_prot)

    temp_model = os.path.join(work_dir, "temp_model1.pdb")
    temp_prot_raw = os.path.join(work_dir, "temp_prot.raw")
    temp_rna_raw = os.path.join(work_dir, "temp_rna.raw")
    temp_prot_renum = os.path.join(work_dir, "temp_prot.renum")
    dssp_file = os.path.join(work_dir, "temp.dssp")
    awk_fix_script = os.path.join(work_dir, "fix_cols.awk")
    awk_parse_ss = os.path.join(work_dir, "parse_ss.awk")

    try:
        cmd_extract = f"awk '/^MODEL/{{p=1}} p{{print}} /^ENDMDL/{{exit}}' '{input_pdb}' > '{temp_model}'"
        subprocess.run(cmd_extract, shell=True, check=True)
        if os.path.getsize(temp_model) < 100:
            subprocess.run(f"cp '{input_pdb}' '{temp_model}'", shell=True, check=True)

        PROT_NAMES = {"ALA","ARG","ASN","ASP","CYS","GLN",
                      "GLU","GLY","HIS","ILE","LEU","LYS",
                      "MET","PHE","PRO","SER","THR","TRP",
                      "TYR","VAL","HIP","HIE","HID","HSP",
                      "CYX","GLH","ASH","LYN"}
        
        RNA_NAMES = {"A","G","C","U","DA","DG","DC","DT",
                     "RA","RG","RC","RU","ADE","GUA","CYT",
                     "URA","THY"}

        with open(temp_model, 'r') as f_in, open(temp_prot_raw, 'w') as f_p, open(temp_rna_raw, 'w') as f_r:
            for line in f_in:
                if line.startswith(("ATOM", "HETATM")):
                    res = line[17:20].strip().upper()
                    if res in PROT_NAMES: f_p.write(line)
                    elif res in RNA_NAMES: f_r.write(line)

        current_old_id = None
        new_id = 0
        with open(temp_prot_raw, 'r') as f_in, open(temp_prot_renum, 'w') as f_out:
            for line in f_in:
                old_id = line[22:26].strip()
                if old_id != current_old_id:
                    new_id += 1
                    current_old_id = old_id

                res = line[17:20].strip().upper()
                if res in ["HIP", "HIE", "HID", "HSP"]: res = "HIS"
                elif res == "CYX": res = "CYS"
                elif res == "GLH": res = "GLU"
                elif res == "ASH": res = "ASP"
                elif res == "LYN": res = "LYS"

                prefix = line[:17] + f"{res:3}" + line[20:22] + f"{new_id:4}" + line[26:54]
                atom_name = line[12:16].replace(" ", "")
                elem = "".join([c for c in atom_name if c.isalpha()])[0]

                f_out.write(f"{prefix:54}  1.00 25.00          {elem:>2}\n")

        header = "HEADER    PREDICTED MODEL ESMFOLD/HDOCK      14-NOV-25   PROT   \n"
        cryst1 = "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1\n"
        with open(output_prot, 'w') as f:
            f.write(header + cryst1)
            with open(temp_prot_renum, 'r') as body: f.write(body.read())
            f.write("ENDMDL\n")

        with open(temp_rna_raw, 'r') as f_in, open(output_rna, 'w') as f_out:
            for line in f_in:
                f_out.write(f"{line[:54]:54}  1.00 25.00\n")
            f_out.write("END\n")

        subprocess.run(f"dssp '{output_prot}' '{dssp_file}'", shell=True, check=True)

        awk_ss_content = r"""
BEGIN { start=0 }
/^  #  RESIDUE/ { start=1; next }
start==1 {
    if (substr($0, 14, 1) == "!") next; # Пропускаем разрывы
    ss = substr($0, 17, 1);
    if (index("HBEGITS", ss) == 0) ss = "C";
    printf "%s", ss
}
"""
        with open(awk_parse_ss, "w") as f: f.write(awk_ss_content)
        subprocess.run(f"awk -f '{awk_parse_ss}' '{dssp_file}' > '{output_ss}'", shell=True, check=True)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    finally:
        for f in [temp_model, temp_prot_raw, temp_rna_raw, temp_prot_renum, dssp_file, awk_fix_script, awk_parse_ss]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    run_strict_prep(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
