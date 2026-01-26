import sys

def merge(input_itp, output_itp):
    found = False
    with open(input_itp, 'r') as f_in, open(output_itp, 'w') as f_out:
        for line in f_in:
            if line.strip().startswith('[ moleculetype ]') or line.strip().startswith('[moleculetype]'):
                found = True
                f_out.write(line)
                continue
            
            if found:
                parts = line.split()
                if len(parts) >= 1:
                    n_atoms = parts[1] if len(parts) > 1 else "1"
                    f_out.write(f"protein       {n_atoms}\n")
                    found = False
                else:
                    f_out.write(line)
            else:
                f_out.write(line)

if __name__ == "__main__":
    merge(sys.argv[1], sys.argv[2])
