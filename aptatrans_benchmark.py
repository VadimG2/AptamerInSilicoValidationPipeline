# ЗАПУСК APTATRANS (Финальная версия с кешированием и всеми исправлениями)
# ==============================================================================
import sys
import numpy as np
import os
import pandas as pd
import subprocess
from tqdm import tqdm
import torch
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from apipred_benchmark import add_roc_auc_to_results, get_reverse_complement, extract_apipred_features_real


from config import (
    DATASET_TYPE, TRAIN_DATA_PATH, TEST_DATA_PATH,
    APIPRED_DIR, APTATRANS_DIR
)

def run_aptatrans_benchmark(test_df):
    """
    Полный цикл для AptaTrans с кешированием вероятностей и расчетом ROC AUC.
    """
    print("\n--- Запуск бенчмарка для AptaTrans ---")

    aptatrans_results_path = os.path.join(APTATRANS_DIR, f'results_{DATASET_TYPE}.csv')
    
    # --- ИЗМЕНЕНИЕ: Логика чтения из кеша ---
    if os.path.exists(aptatrans_results_path):
        print(f"Загрузка готовых результатов AptaTrans из: {aptatrans_results_path}")
        results_df = pd.read_csv(aptatrans_results_path)
        y_test = results_df['true_labels'].values
        predicted_labels = results_df['predictions'].values
        
        report = classification_report(y_test, predicted_labels, output_dict=True)
        accuracy = accuracy_score(y_test, predicted_labels)
        
        results = {'accuracy': accuracy, 'report': report, 'predictions': predicted_labels, 'true_labels': y_test}

        # Пытаемся загрузить вероятности для ROC AUC
        if 'probabilities' in results_df.columns:
            y_pred_proba = results_df['probabilities'].values
            results = add_roc_auc_to_results(results, y_test, y_pred_proba)
        else:
            print("В кеше отсутствуют вероятности, ROC AUC не рассчитан.")
            results['roc_auc'] = 0.0

        print("--- Бенчмарк AptaTrans завершен (результаты загружены из кеша) ---")
        return results

    # --- ИЗМЕНЕНИЕ: Логика записи в кеш ---
    print("Файл с результатами не найден. Запуск полного цикла предсказаний AptaTrans...")
    original_working_dir = os.getcwd()
    try:
        print(f"Временно изменяю рабочую директорию на: {APTATRANS_DIR}")
        os.chdir(APTATRANS_DIR)
        
        sys.path.insert(0, '.')
        from aptatrans_pipeline import AptaTransPipeline 
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"AptaTrans: используется устройство: {DEVICE}")

        print("AptaTrans: Инициализация пайплайна...")
        pipeline = AptaTransPipeline(dim=128, mult_ff=2, n_layers=6, n_heads=8, dropout=0.1, load_best_pt=True, device=DEVICE, seed=1004)

        successful_probabilities = [] # Переименовано для ясности
        successful_true_labels = []

        print("AptaTrans: Получение предсказаний...")
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="AptaTrans Inference", mininterval=1.0):
            if not isinstance(row['apt_seq'], str) or not isinstance(row['target_seq'], str):
                continue
            try:
                score = pipeline.inference(row['apt_seq'], row['target_seq'])
                successful_probabilities.append(score)
                successful_true_labels.append(row['label'])
            except Exception as e:
                continue

        if not successful_probabilities:
            print("\n\n!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось получить ни одного предсказания от AptaTrans. !!!")
            return {'accuracy': 0.0, 'report': {}, 'predictions': [], 'true_labels': []}

        threshold = 0.5
        y_pred_proba_flat = np.array(successful_probabilities).ravel()
        predicted_labels_flat = (y_pred_proba_flat >= threshold).astype(int)
        y_test_flat = np.array(successful_true_labels).ravel()
        
        print(f"Сохранение результатов AptaTrans в: {aptatrans_results_path}")
        # Сохраняем и предсказания, и вероятности
        results_to_save = pd.DataFrame({
            'true_labels': y_test_flat, 
            'predictions': predicted_labels_flat,
            'probabilities': y_pred_proba_flat
        })
        results_to_save.to_csv(aptatrans_results_path, index=False)
        
        report = classification_report(y_test_flat, predicted_labels_flat, output_dict=True)
        accuracy = accuracy_score(y_test_flat, predicted_labels_flat)
        
        results = {
            'accuracy': accuracy, 
            'report': report, 
            'predictions': predicted_labels_flat, 
            'true_labels': y_test_flat
        }
        
        # Добавляем ROC AUC
        results = add_roc_auc_to_results(results, y_test_flat, y_pred_proba_flat)
        
        print("--- Бенчмарк AptaTrans завершен ---")
        return results

    finally:
        print(f"Возвращаю рабочую директорию на: {original_working_dir}")
        os.chdir(original_working_dir)
