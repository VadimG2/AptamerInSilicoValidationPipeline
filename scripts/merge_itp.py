import sys
import os

def clean_and_rename_itp(input_file, output_file, new_name="protein"):
    """
    Берет файл molecule_0.itp, удаляет из него инклюды мартини и системные блоки,
    и меняет имя в [ moleculetype ] на заданное.
    """
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found")
        sys.exit(1)

    lines = []
    with open(input_file, 'r') as f:
        lines = f.readlines()

    final_lines = []
    skip_block = False
    found_moleculetype = False

    for line in lines:
        if line.startswith("#include") or line.startswith("[ system ]") or line.startswith("[ molecules ]"):
            continue

        if line.strip().startswith("[ moleculetype ]"):
            final_lines.append(line)
            found_moleculetype = True
            continue

        if found_moleculetype:
            if line.strip() and not line.strip().startswith(";"):
                parts = line.split()
                parts[0] = new_name
                final_lines.append(f"{parts[0]:10} {parts[1] if len(parts)>1 else '3'}\n")
                found_moleculetype = False
                continue

        final_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(final_lines)
    print(f"SUCCESS: {input_file} converted to {output_file} with name '{new_name}'")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_itp.py <input_itp> <output_itp>")
        sys.exit(1)
    clean_and_rename_itp(sys.argv[1], sys.argv[2])
