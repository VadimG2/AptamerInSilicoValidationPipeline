import pandas as pd
import subprocess
import torch
from tqdm import tqdm
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from repDNA.nac import Kmer
import pepfeature as pep
import os
import re
from sklearn.metrics import roc_auc_score

from config import (
    DATASET_TYPE, TRAIN_DATA_PATH, TEST_DATA_PATH,
    APIPRED_DIR, APTATRANS_DIR
)


def add_roc_auc_to_results(results, y_true, y_pred_proba):
    """Добавляет ROC AUC в словарь с результатами."""
    try:
        # Убедимся, что y_pred_proba - это вероятности положительного класса
        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]
            
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        results['roc_auc'] = roc_auc
        print(f"ROC AUC: {roc_auc:.4f}")
    except Exception as e:
        print(f"Не удалось посчитать ROC AUC: {e}")
        results['roc_auc'] = 0.0
    return results

def get_reverse_complement(seq):
    """
    Returns the reverse complement of a DNA sequence. Assumes input is already in DNA format (ACGT).
    """
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement_dict.get(base, base) for base in seq[::-1])

def extract_apipred_features_real(df_input):
    """
    Extracts features with robust cleaning of input sequences and manual normalization for repDNA.
    """
    print(f"Extracting features for {len(df_input)} pairs...")
    kmer = Kmer(k=3)
    all_feature_vectors = []
    successful_indices = []
    protein_feature_columns = None
    
    for index, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Extracting APIPred features", mininterval=1.0):
        try:
            apt_seq = row['apt_seq']
            prot_seq = row['target_seq']
            
            # Очистка и подготовка последовательности аптамера
            apt_seq_cleaned = re.sub(r'[^ACGU]', '', str(apt_seq).upper())
            apt_seq_dna_format = apt_seq_cleaned.replace('U', 'T')
            
            if len(apt_seq_dna_format) < 3:
                continue
            
            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ: Ручная нормализация ---
            # Расчет прямого k-mer состава
            kmer_counts = kmer.make_kmer_vec([apt_seq_dna_format])[0]
            kmer_sum = np.sum(kmer_counts)
            kmer_freq = kmer_counts / kmer_sum if kmer_sum > 0 else kmer_counts
            
            # Расчет обратно-комплементарного k-mer состава
            rev_seq = get_reverse_complement(apt_seq_dna_format)
            rev_kmer_counts = kmer.make_kmer_vec([rev_seq])[0]
            rev_kmer_sum = np.sum(rev_kmer_counts)
            rev_kmer_freq = rev_kmer_counts / rev_kmer_sum if rev_kmer_sum > 0 else rev_kmer_counts
            
            aptamer_vector = np.concatenate([kmer_freq, rev_kmer_freq])
            
            # Очистка и обработка последовательности белка
            prot_seq_cleaned = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', str(prot_seq).upper())
            if not prot_seq_cleaned:
                continue

            temp_df = pd.DataFrame({'target_seq': [prot_seq_cleaned]})
            try:
                protein_features_df = pep.aa_descriptors.calc_df(
                    dataframe=temp_df,
                    aa_column='target_seq',
                    Ncores=1
                )
                if protein_features_df is None or protein_features_df.empty:
                    raise ValueError("pepfeature вернул пустой результат")
            except Exception as e:
                print(f"pepfeature упал на белке длиной {len(prot_seq_cleaned)}. Ошибка: {e}. Пропуск.")
                continue

            # --- Извлечение колонок ---
            if protein_feature_columns is None:
                protein_feature_columns = [col for col in protein_features_df.columns if col != 'target_seq']
            
            protein_vector = np.array([protein_features_df[col].iloc[0] for col in protein_feature_columns])

            # Сборка финального вектора
            combined_vector = np.concatenate([aptamer_vector, protein_vector])
            all_feature_vectors.append(combined_vector)
            successful_indices.append(index)

        except Exception as e:
            # Отладка на случай непредвиденных ошибок
            print(f"\nCaught unhandled exception. Aptamer: {row.get('apt_seq', 'N/A')}, Protein: {row.get('target_seq', 'N/A')}. Error: {e}. Skipping.")
            continue

    if not successful_indices:
        raise ValueError("No sequences were successfully processed. Check data quality (e.g., length, content, non-standard characters).")
    
    successful_labels = df_input.loc[successful_indices, 'label'].values
    feature_df = pd.DataFrame(all_feature_vectors)
    feature_df.columns = [f'feature_{i}' for i in range(feature_df.shape[1])]
    feature_df.insert(0, 'Class', successful_labels)
    
    print(f"Feature extraction completed. Processed {len(successful_indices)} out of {len(df_input)} pairs.")
    return feature_df

# ... (остальные функции run_apipred_pipeline и run_apipred_benchmark остаются без изменений) ...

def run_apipred_pipeline(train_features_path, test_features_path):
    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.feature_selection import SelectFromModel
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.preprocessing import PolynomialFeatures
    train = pd.read_csv(train_features_path)
    test = pd.read_csv(test_features_path)
    X_train, y_train = train.drop(columns=['Class']), train['Class']
    X_test, y_test = test.drop(columns=['Class']), test['Class']
    X_train_np, X_test_np = X_train.values, X_test.values
    print("APIPred: Запуск отбора признаков...")
    clf_fs = XGBClassifier(eval_metric='logloss', random_state=42)
    clf_fs.fit(X_train_np, y_train)
    selection = SelectFromModel(clf_fs, threshold=0.01, prefit=True)
    X_train_selected = selection.transform(X_train_np)
    X_test_selected = selection.transform(X_test_np)
    print("APIPred: Создание полиномиальных признаков...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_selected)
    X_test_poly = poly.transform(X_test_selected)
    print("APIPred: Балансировка классов (RandomOverSampler)...")
    sampler = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = sampler.fit_resample(X_train_poly, y_train)
    print("APIPred: Обучение модели XGBoost с подбором параметров...")
    clf = XGBClassifier(random_state=42)
    parameters = {'n_estimators': [200], 'learning_rate': [0.1], 'max_depth': [10], 'eval_metric': ['logloss']}
    clf_gs = GridSearchCV(clf, parameters, cv=3, n_jobs=1)
    clf_gs.fit(X_train_res, y_train_res)
    print("APIPred: Получение предсказаний...")
    y_pred_proba = clf_gs.predict_proba(X_test_poly)
    y_pred = (y_pred_proba[:,1] >= 0.4).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    results = {'accuracy': accuracy, 'report': report, 'predictions': y_pred, 'true_labels': y_test.values}
    results = add_roc_auc_to_results(results, y_test, y_pred_proba)
    return results

def run_apipred_benchmark(train_df, test_df):
    """
    Полный цикл для APIPred с правильной логикой кеширования:
    - Проверяет наличие файлов с признаками.
    - Загружает, если они есть.
    - Генерирует и сохраняет, если их нет.
    """
    print("\n--- Запуск бенчмарка для APIPred ---")
    print("Шаг 1/3: Проверка/Извлечение признаков...")
    
    # Определяем пути к файлам с признаками для текущего DATASET_TYPE
    apipred_train_path = os.path.join(APIPRED_DIR, f'train_Dataset_{DATASET_TYPE}.csv')
    apipred_test_path = os.path.join(APIPRED_DIR, f'test_Dataset_{DATASET_TYPE}.csv')
    
    # --- Обработка признаков для ОБУЧАЮЩЕГО набора ---
    if os.path.exists(apipred_train_path):
        print(f"Найден готовый файл с признаками для обучающего набора: {apipred_train_path}")
    else:
        print(f"Файл с признаками для обучающего набора не найден. Запуск извлечения...")
        train_features_df = extract_apipred_features_real(train_df)
        train_features_df.to_csv(apipred_train_path, index=False)
        print(f"Признаки для обучающего набора сохранены в {apipred_train_path}")

    # --- Обработка признаков для ТЕСТОВОГО набора ---
    if os.path.exists(apipred_test_path):
        print(f"Найден готовый файл с признаками для тестового набора: {apipred_test_path}")
    else:
        print(f"Файл с признаками для тестового набора не найден. Запуск извлечения...")
        test_features_df = extract_apipred_features_real(test_df)
        test_features_df.to_csv(apipred_test_path, index=False)
        print(f"Признаки для тестового набора сохранены в {apipred_test_path}")

    # Шаг 2 теперь не нужен, так как файлы уже на месте
    print("\nШаг 3/3: Запуск обучения и предсказания...")
    apipred_results = run_apipred_pipeline(apipred_train_path, apipred_test_path)
    
    print("--- Бенчмарк APIPred завершен ---")
    return apipred_results

