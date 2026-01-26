import subprocess
from pathlib import Path
from tqdm import tqdm
import pandas as pd

docking_dir = Path("./docking_candidates")
candidates_csv_path = docking_dir / "candidates_for_docking.csv"

# Проверка входных данных
if not candidates_csv_path.exists():
    print(f"[ОШИБКА] Файл {candidates_csv_path} не найден. Запустите блок 1.")
    raise SystemExit(1)

candidates_df = pd.read_csv(candidates_csv_path)
print(f"Начинаем предсказание 2D-структур для {len(candidates_df)} аптамеров...")

successful_count = 0
failed_count = 0

for _, row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="RNAfold"):
    candidate_id = row['candidate_id']
    candidate_dir = docking_dir / candidate_id
    aptamer_fasta = candidate_dir / "aptamer.fasta"
    aptamer_ss = candidate_dir / "aptamer.ss"

    if aptamer_ss.exists():
        successful_count += 1
        continue

    if not aptamer_fasta.exists():
        print(f"  [Пропуск] {candidate_id}: aptamer.fasta не найден")
        failed_count += 1
        continue
    try:
        with open(aptamer_fasta, 'r') as f:
            lines = f.readlines()
            seq = "".join(line.strip() for line in lines if not line.startswith('>'))
    except Exception as e:
        print(f"  [ОШИБКА] Чтение {aptamer_fasta}: {e}")
        failed_count += 1
        continue

    if not seq:
        print(f"  [Пропуск] {candidate_id}: пустая последовательность")
        failed_count += 1
        continue

    # DNA → RNA
    rna_seq = seq.upper().replace('T', 'U')

    # Подготовка входа для RNAfold
    input_data = f">{candidate_id}\n{rna_seq}\n"

    try:
        result = subprocess.run(
            ["RNAfold", "--noPS"],
            input=input_data,
            capture_output=True,
            check=True,
            text=True,
            timeout=60
        )

        with open(aptamer_ss, "w") as f:
            f.write(result.stdout)

        successful_count += 1

    except FileNotFoundError:
        print("\n[КРИТИЧЕСКАЯ ОШИБКА] RNAfold не найден в PATH!")
        print("   Установите: conda install -c bioconda viennarna")
        print("   Или: sudo apt install viennarna")
        raise SystemExit(1)

    except subprocess.TimeoutExpired:
        print(f"  [ОШИБКА] {candidate_id}: RNAfold завис (timeout)")
        with open(candidate_dir / "APTAMER_ERROR.txt", "w") as f:
            f.write("RNAfold timeout\n")
        failed_count += 1

    except subprocess.CalledProcessError as e:
        print(f"  [ОШИБКА] {candidate_id}: RNAfold ошибка")
        with open(candidate_dir / "APTAMER_ERROR.txt", "w") as f:
            f.write(f"RNAfold error:\n{e.stderr}\n")
        failed_count += 1

    except Exception as e:
        print(f"  [НЕИЗВЕСТНАЯ ОШИБКА] {candidate_id}: {e}")
        failed_count += 1

# Итог
print("\n" + "="*60)
print("ПРЕДСКАЗАНИЕ 2D-СТРУКТУР ЗАВЕРШЕНО")
print(f"  Успешно: {successful_count}")
print(f"  Пропущено (уже есть): {len(candidates_df) - successful_count - failed_count}")
print(f"  Ошибки: {failed_count}")
print("="*60)

with open("block3.done", "w") as f:
    f.write("2D structures predicted\n")
print("block3.done создан — блок 3 завершён.")
