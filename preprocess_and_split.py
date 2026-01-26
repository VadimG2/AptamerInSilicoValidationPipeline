import pandas as pd
import numpy as np
import random
import os
import subprocess
from pathlib import Path
import re

random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# Загрузка df
if 'df' not in locals():
    df = pd.read_csv('Aptamers_and_targets.csv')
    print("df загружен из CSV.")

df.info()


# Шаг 1: Предобработка
print("1. Предобработка данных...")

core_cols = ['apt_seq', 'target_seq']
positive_df = df[core_cols].copy()

positive_df.drop_duplicates(inplace=True)
positive_df.dropna(inplace=True)
positive_df['label'] = 1

print(f"Осталось {len(positive_df)} уникальных положительных пар после очистки.")
print("-" * 50)

# Шаг 2: Негативная выборка с прогрессом
print("2. Создание негативной выборки...")

unique_apts = positive_df['apt_seq'].unique().tolist()
unique_targets = positive_df['target_seq'].unique().tolist()

positive_pairs_set = set(zip(positive_df['apt_seq'], positive_df['target_seq']))

negative_samples = []
num_negative_samples = len(positive_df) * 3
iterations = 0

print(f"Генерируем {num_negative_samples} негативных пар (прогресс каждые 1000 итераций)...")

while len(negative_samples) < num_negative_samples:
    apt = random.choice(unique_apts)
    target = random.choice(unique_targets)
    
    if (apt, target) not in positive_pairs_set:
        negative_samples.append({'apt_seq': apt, 'target_seq': target, 'label': 0})
    
    iterations += 1
    if iterations % 1000 == 0:
        print(f"Итерации: {iterations}, Негативных: {len(negative_samples)}")

negative_df = pd.DataFrame(negative_samples).drop_duplicates()

print(f"Сгенерировано {len(negative_df)} уникальных негативных пар.")

combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Итоговый размер датасета: {len(combined_df)} (Положительные: {len(positive_df)}, Отрицательные: {len(negative_df)})")
print("-" * 50)

# Шаг 3: РНК/ДНК
print("3. Разделение на РНК и ДНК датасеты...")

df_rna = combined_df[combined_df['apt_seq'].str.contains('U')].copy()
df_dna = combined_df[~combined_df['apt_seq'].str.contains('U')].copy()

print(f"Найдено {len(df_rna)} пар с РНК-аптамерами.")
print(f"Найдено {len(df_dna)} пар с ДНК-аптамерами.")
print("-" * 50)

# Шаг 4: CD-HIT
print("4. Разделение на Train/Test с помощью кластеризации (CD-HIT)...")

CDHIT_PATH = "./cdhit/cd-hit"  
CDHIT_EST_PATH = "./cdhit/cd-hit-est" 

temp_dir = Path("./cdhit_temp")
temp_dir.mkdir(exist_ok=True)

def run_cd_hit(sequences: list, output_path: Path, command: str, threshold: float, is_protein: bool):
    fasta_path = output_path.with_suffix('.fasta')
    with open(fasta_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")
    
    word_size = 5 if is_protein else 10
    
    cmd_str = f"{command} -i {fasta_path} -o {output_path} -c {threshold} -n {word_size} -d 0 -T 0"
    
    print(f"Запуск команды: {cmd_str}")
    print(f"Прогресс CD-HIT: следите за файлом {output_path}.clstr...")
    try:
        subprocess.run(cmd_str, shell=True, check=True)
        print(f"CD-HIT завершён: {output_path}.clstr создан.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ОШИБКА: Не удалось выполнить CD-HIT. Убедитесь, что он установлен и путь '{command}' верен.")
        print(e)
        return None
    
    return output_path.with_suffix('.clstr')

def parse_clusters(cluster_file: Path):
    clusters = []
    if not cluster_file.exists():
        return clusters
        
    with open(cluster_file, 'r') as f:
        cluster = []
        for line in f:
            if line.startswith(">"):
                if cluster:
                    clusters.append(cluster)
                cluster = []
            else:
                match = re.search(r">seq_(\d+)\.\.\.", line)
                if match:
                    cluster.append(int(match.group(1)))
        if cluster:
            clusters.append(cluster)
    return clusters

def split_by_clusters(sequences: list, command: str, is_protein: bool, threshold=0.85, test_size=0.2):
    output_path = temp_dir / f"{'protein' if is_protein else 'aptamer'}_clusters"
    cluster_file = run_cd_hit(sequences, output_path, command, threshold, is_protein)
    
    if not cluster_file:
        return None, None
        
    clusters = parse_clusters(cluster_file)
    random.seed(42)
    random.shuffle(clusters)
    
    test_indices = []
    train_indices = []
    target_test_count = int(len(sequences) * test_size)
    
    for cluster in clusters:
        if len(test_indices) < target_test_count:
            test_indices.extend(cluster)
        else:
            train_indices.extend(cluster)
            
    train_seqs = {sequences[i] for i in train_indices}
    test_seqs = {sequences[i] for i in test_indices}
    
    return train_seqs, test_seqs

# Разделение ДНК

if not df_dna.empty:
    print("\n--- Обработка ДНК-датасета ---")
    unique_dna_apts = df_dna['apt_seq'].unique().tolist()
    unique_dna_targets = df_dna['target_seq'].unique().tolist()

    print(f"Уникальных ДНК-аптамеров: {len(unique_dna_apts)}")
    print(f"Уникальных белков: {len(unique_dna_targets)}")

    train_apts_dna, test_apts_dna = split_by_clusters(unique_dna_apts, CDHIT_EST_PATH, is_protein=False, threshold=0.85)
    train_targets_dna, test_targets_dna = split_by_clusters(unique_dna_targets, CDHIT_PATH, is_protein=True, threshold=0.85)

    if train_apts_dna is not None:
        test_mask_dna = df_dna['apt_seq'].isin(test_apts_dna) | df_dna['target_seq'].isin(test_targets_dna)
        train_mask_dna = df_dna['apt_seq'].isin(train_apts_dna) & df_dna['target_seq'].isin(train_targets_dna)
        
        train_dna_df = df_dna[train_mask_dna]
        test_dna_df = df_dna[test_mask_dna]

        print("\nРезультаты разделения для ДНК:")
        print(f"Размер обучающей выборки: {len(train_dna_df)}")
        print(f"Размер тестовой выборки: {len(test_dna_df)}")
        
        train_dna_df.to_csv("train_dna_dataset.csv", index=False)
        test_dna_df.to_csv("test_dna_dataset.csv", index=False)
        print("ДНК датасеты сохранены.")

# Разделение РНК
if not df_rna.empty:
    print("\n--- Обработка РНК-датасета ---")
    unique_rna_apts = df_rna['apt_seq'].unique().tolist()
    unique_rna_targets = df_rna['target_seq'].unique().tolist()

    print(f"Уникальных РНК-аптамеров: {len(unique_rna_apts)}")
    print(f"Уникальных белков: {len(unique_rna_targets)}")

    train_apts_rna, test_apts_rna = split_by_clusters(unique_rna_apts, CDHIT_EST_PATH, is_protein=False, threshold=0.85)
    train_targets_rna, test_targets_rna = split_by_clusters(unique_rna_targets, CDHIT_PATH, is_protein=True, threshold=0.85)
    
    if train_apts_rna is not None:
        test_mask_rna = df_rna['apt_seq'].isin(test_apts_rna) | df_rna['target_seq'].isin(test_targets_rna)
        train_mask_rna = df_rna['apt_seq'].isin(train_apts_rna) & df_rna['target_seq'].isin(train_targets_rna)
        
        train_rna_df = df_rna[train_mask_rna]
        test_rna_df = df_rna[test_mask_rna]

        print("\nРезультаты разделения для РНК:")
        print(f"Размер обучающей выборки: {len(train_rna_df)}")
        print(f"Размер тестовой выборки: {len(test_rna_df)}")
        
        train_rna_df.to_csv("train_rna_dataset.csv", index=False)
        test_rna_df.to_csv("test_rna_dataset.csv", index=False)
        print("РНК датасеты сохранены в train_rna_dataset.csv и test_rna_dataset.csv")

print("=== ПРЕДОБРАБОТКА ЗАВЕРШЕНА ===")
