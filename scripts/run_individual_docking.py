import subprocess
import os
from pathlib import Path
import sys
import shutil
import tempfile

def is_pdb_valid(filepath):
    if not filepath.exists() or filepath.stat().st_size == 0:
        return False
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                return True
    return False

input_receptor = Path(sys.argv[1]).resolve()
input_ligand = Path(sys.argv[2]).resolve()
output_score_file = Path(sys.argv[3]).resolve()
output_complex_pdb = Path(sys.argv[4]).resolve()
hdock_path = Path(sys.argv[5]).resolve()
createpl_path = Path(sys.argv[6]).resolve()
nmax_config = int(sys.argv[7])
candidate_id = sys.argv[8]
model_name = sys.argv[9]

output_score_file.parent.mkdir(parents=True, exist_ok=True)
current_env = os.environ.copy()

custom_fftw_path = Path("/mnt/tank/scratch/vgvozdev/apta/resources/fftw/lib")


current_env["LD_LIBRARY_PATH"] = f"{custom_fftw_path}:{current_env.get('LD_LIBRARY_PATH', '')}"

try:
    with tempfile.TemporaryDirectory(prefix=f"dock_{model_name}_") as tmpdirname:
        work_dir = Path(tmpdirname)
        print(f"Working in isolated temp directory: {work_dir}")

        local_hdock_out = work_dir / "hdock.out"

        if not is_pdb_valid(input_receptor):
            print(f"Error: Invalid receptor: {input_receptor}", file=sys.stderr)
            sys.exit(1)
        if not is_pdb_valid(input_ligand):
            print(f"Error: Invalid ligand: {input_ligand}", file=sys.stderr)
            output_score_file.touch(); output_complex_pdb.touch()
            sys.exit(0)

        pdb4amber = shutil.which("pdb4amber")
        if not pdb4amber:
            print("Error: pdb4amber not found", file=sys.stderr); sys.exit(1)

        try:
            subprocess.run([pdb4amber, "-i", str(input_receptor), "-o", "receptor_clean.pdb"],
                           cwd=work_dir, capture_output=True, check=True, env=current_env)
            subprocess.run([pdb4amber, "-i", str(input_ligand), "-o", "ligand_clean.pdb"],
                           cwd=work_dir, capture_output=True, check=True, env=current_env)
        except subprocess.CalledProcessError:
            print("Error: pdb4amber failed", file=sys.stderr); sys.exit(1)

        print("Running HDOCK...")
        cmd_hdock = [str(hdock_path), "receptor_clean.pdb", "ligand_clean.pdb", "-out", "hdock.out"]

        proc = subprocess.run(cmd_hdock, cwd=work_dir, capture_output=True, text=True, env=current_env)

        if proc.returncode != 0 or not local_hdock_out.exists():
            print(f"HDOCK failed. Stderr: {proc.stderr}", file=sys.stderr)
            print(f"Debug: LD_LIBRARY_PATH was: {current_env['LD_LIBRARY_PATH']}", file=sys.stderr)
            sys.exit(1)

        best_score = None
        solutions = 0
        with open(local_hdock_out, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 7 and parts[0].replace('.','',1).isdigit():
                    if best_score is None: best_score = parts[6]
                    solutions += 1

        if best_score is None:
            print("No solutions found.", file=sys.stderr)
            output_score_file.touch(); output_complex_pdb.touch()
            sys.exit(0)

        with open(output_score_file, 'w') as f:
            f.write(str(best_score))

        n_models = min(nmax_config, solutions)
        print(f"Generating {n_models} models...")

        cmd_createpl = [str(createpl_path), "hdock.out", "complex.pdb", "-nmax", str(n_models), "-complex", "-models"]
        subprocess.run(cmd_createpl, cwd=work_dir, capture_output=True, check=True, env=current_env)

        model_files = sorted(work_dir.glob("model_*.pdb"), key=lambda x: int(x.stem.split('_')[1]))

        if not model_files:
            print("Error: No model files created", file=sys.stderr); sys.exit(1)

        with open(output_complex_pdb, 'wb') as outfile:
            for i, mfile in enumerate(model_files):
                outfile.write(f"MODEL        {i+1}\n".encode())
                with open(mfile, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
                outfile.write(b"ENDMDL\n")

    print(f"Success for {model_name}")

except Exception as e:
    print(f"Critical error: {e}", file=sys.stderr)
    sys.exit(1)
