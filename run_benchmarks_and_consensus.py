from sklearn.metrics import confusion_matrix
import numpy as np
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH,
    APIPRED_DIR, APTATRANS_DIR, APTANET_DIR,
    DOCKING_DIR, DATASET_TYPE
)

from apipred_benchmark import run_apipred_benchmark
from aptatrans_benchmark import run_aptatrans_benchmark
from aptanet_benchmark import run_aptanet_benchmark

# Загрузка данных
print("Загрузка данных...")
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
print(f"Загружено {len(train_df)} обучающих и {len(test_df)} тестовых примеров.")

# Запуск бенчмарков
all_benchmark_results = {}

# 1. Запуск APIPred
try:
    results_apipred = run_apipred_benchmark(train_df, test_df)
    all_benchmark_results['APIPred'] = results_apipred
except Exception as e:
    print(f"\n!!! ОШИБКА при выполнении бенчмарка APIPred: {e} !!!")

# 2. Запуск AptaTrans
try:
    results_aptatrans = run_aptatrans_benchmark(test_df)
    all_benchmark_results['AptaTrans'] = results_aptatrans
except Exception as e:
    print(f"\n!!! ОШИКА при выполнении бенчмарка AptaTrans: {e} !!!")

# 3. AptaNet
try:
    results_aptanet = run_aptanet_benchmark(train_df, test_df)
    all_benchmark_results['AptaNet'] = results_aptanet
except Exception as e:
    print(f"AptaNet ошибка: {e}")
    import traceback
    traceback.print_exc()

# Создание сводной таблицы
summary_data = []
for model_name, results in all_benchmark_results.items():
    if not results or not results.get('report'):
        continue

    # Извлекаем основные метрики, используя .get() для безопасности
    accuracy = results.get('accuracy', 0.0)
    roc_auc = results.get('roc_auc', 0.0)
    
    # Извлекаем метрики для класса "1" (связывается)
    report_class_1 = results.get('report', {}).get('1', {})
    precision_bind = report_class_1.get('precision', 0.0)
    recall_bind = report_class_1.get('recall', 0.0)
    f1_score_bind = report_class_1.get('f1-score', 0.0)
    
    summary_data.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'ROC AUC': roc_auc,
        'Precision (Bind)': precision_bind,
        'Recall (Bind)': recall_bind,
        'F1-score (Bind)': f1_score_bind
    })

# Создаем и форматируем DataFrame
summary_df = pd.DataFrame(summary_data)
summary_df.set_index('Model', inplace=True)
print('В AptaNet DNN заменен на XGBoost!')
print(summary_df.round(4))

# 1. Сохранение в текстовый файл
summary_txt_path = 'benchmark_summary.txt'
print(f"Сохранение текстового отчета в: {summary_txt_path}")
with open(summary_txt_path, 'w') as f:
    for model_name, results in all_benchmark_results.items():
        if not results or not results.get('report'):
            f.write(f"--- {model_name} ---\n")
            f.write("Результаты отсутствуют или некорректны.\n")
            f.write("--------------------\n\n")
            continue
        
        f.write(f"--- {model_name} ---\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
        
        f.write("Classification Report:\n")
        report_df = pd.DataFrame(results['report']).transpose()
        f.write(report_df[['precision', 'recall', 'f1-score', 'support']].round(4).to_string())
        f.write("\n\n")
        
        f.write("Confusion Matrix:\n")
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        f.write(np.array2string(cm))
        f.write("\n--------------------\n\n")

# 2. Сохранение в JSON файл
summary_json_path = 'benchmark_summary.json'
results_for_json = {}
for model_name, results in all_benchmark_results.items():
    if results and results.get('report'):
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        results_for_json[model_name] = {
            'accuracy': results['accuracy'],
            'classification_report': results['report'],
            'confusion_matrix': cm.tolist()
        }

print(f"Сохранение JSON отчета в: {summary_json_path}")
with open(summary_json_path, 'w') as f:
    json.dump(results_for_json, f, indent=4)

print("\nВсе результаты успешно сохранены.")


# Блок 5: Подготовка консенсусных кандидатов по трем моделям

# Проверка наличия исходного DataFrame
if 'df' not in locals():
    print("Загрузка исходного полного DataFrame 'Aptamers_and_targets.csv'...")
    df = pd.read_csv('Aptamers_and_targets.csv')
    df = df[['apt_seq', 'target_seq', 'target_name']].drop_duplicates(subset=['apt_seq', 'target_seq'])
else:
    print(f"Исходный DataFrame 'df' уже загружен ({len(df)} уникальных пар).")

# Проверка результатов
required_results = ['APIPred', 'AptaTrans', 'AptaNet']
missing = [name for name in required_results if name not in all_benchmark_results]
if missing:
    print(f"ОШИБКА: Отсутствуют результаты для моделей: {', '.join(missing)}")
    print("Запустите предыдущие ячейки с бенчмарками.")
else:
    print(f"Все три модели найдены: {', '.join(required_results)}")

    # Шаг 1: Подготовка test_df с target_name
    enriched_test_df = pd.merge(test_df, df, on=['apt_seq', 'target_seq'], how='left')
    if 'target_name' not in enriched_test_df.columns:
        print("Критическая ошибка: не удалось добавить target_name.")
    else:
        print(f"Обогащённый test_df: {len(enriched_test_df)} строк.")

    # Шаг 2: Создание DataFrame'ов с предсказаниями
    dfs = {}

    for model_name in required_results:
        results = all_benchmark_results[model_name]
        pred_key = 'predictions'
        proba_key = 'probabilities'

        if pred_key not in results or len(results[pred_key]) != len(enriched_test_df):
            print(f"ОШИБКА: Несоответствие длины предсказаний {model_name}: {len(results[pred_key])} vs {len(enriched_test_df)}")
            continue

        df_model = enriched_test_df.copy()
        df_model = df_model.iloc[:len(results[pred_key])].reset_index(drop=True)
        df_model['prediction'] = results[pred_key]

        # Добавляем вероятности P(bind)
        if proba_key in results and len(results[proba_key]) == len(df_model):
            df_model['proba_1'] = results[proba_key]
        else:
            df_model['proba_1'] = df_model['prediction'].astype(float)

        # Только положительные
        df_pos = df_model[df_model['prediction'] == 1].copy()
        df_pos = df_pos[['apt_seq', 'target_seq', 'target_name', 'proba_1']].rename(columns={'proba_1': f'proba_1_{model_name.lower()}'})
        dfs[model_name] = df_pos
        print(f"{model_name}: {len(df_pos)} положительных предсказаний.")

    # Шаг 3: Консенсус
    if len(dfs) != 3:
        print("Недостаточно моделей для консенсуса.")
    else:
        consensus = dfs['APIPred']
        for model in ['AptaTrans', 'AptaNet']:
            consensus = pd.merge(
                consensus, dfs[model],
                on=['apt_seq', 'target_seq', 'target_name'],
                how='inner'
            )
        consensus = pd.merge(
            consensus,
            enriched_test_df[['apt_seq', 'target_seq', 'label']],
            on=['apt_seq', 'target_seq'],
            how='left'
        )
        consensus = consensus.rename(columns={'label': 'true_label'})

        print(f"\nКонсенсус по ТРЁМ моделям: {len(consensus)} кандидатов.")

        if not consensus.empty:
            # Шаг 4: Подготовка файлов
            docking_dir = Path("./docking_candidates")
            docking_dir.mkdir(exist_ok=True)
            print(f"\nДиректория для докинга: {docking_dir}")

            csv_path = docking_dir / "candidates_for_docking.csv"
            consensus.to_csv(csv_path, index=False)
            print(f"Список кандидатов сохранён: {csv_path}")

            # Создаём candidate_id
            consensus = consensus.reset_index(drop=True)
            consensus['candidate_id'] = [f'candidate_{i+1:03d}' for i in range(len(consensus))]

            print("\nСоздание FASTA-файлов...")
            for _, row in tqdm(consensus.iterrows(), total=len(consensus), desc="Файлы кандидатов"):
                cid = row['candidate_id']
                cdir = docking_dir / cid
                cdir.mkdir(exist_ok=True)

                seq_to_save = row['apt_seq'].upper()
                if 'DATASET_TYPE' in globals() and DATASET_TYPE == 'rna':
                    seq_to_save = seq_to_save.replace('T', 'U')

                with open(cdir / "aptamer.fasta", "w") as f:
                    f.write(f">{cid}_{row['target_name']}_aptamer\n{seq_to_save}\n")

                with open(cdir / "protein.fasta", "w") as f:
                    f.write(f">{cid}_{row['target_name']}_protein\n{row['target_seq'].upper()}\n")

            print(f"\nГОТОВО! {len(consensus)} консенсусных кандидатов подготовлены.")
            print(f"→ Следующий шаг: Блок 6 (UniProt + ESMFold)")

        else:
            print("Консенсусных кандидатов не найдено.")
