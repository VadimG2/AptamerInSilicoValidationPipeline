# БЛОК: AptaNet — ФИНАЛЬНАЯ ВЕРСИЯ (XGBoost + ПОЛНОЕ ВЫРАВНИВАНИЕ + ЗАЩИТА)
# ==============================================================================
from config import (
    DATASET_TYPE, TRAIN_DATA_PATH, TEST_DATA_PATH,
    APIPRED_DIR, APTATRANS_DIR, BASE_DIR
)

def run_aptanet_benchmark(train_df, test_df):
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from collections import Counter
    from sklearn.feature_selection import SelectFromModel
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
    import os
    import joblib
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    import itertools

    print("\n--- Запуск бенч-марка для AptaNet (по статье) ---")

    APTANET_DIR = os.path.join(BASE_DIR, 'AptaNet')
    os.makedirs(APTANET_DIR, exist_ok=True)
    model_path = os.path.join(APTANET_DIR, 'aptanet_model.pkl')
    selector_path = os.path.join(APTANET_DIR, 'feature_selector.pkl')

    # --- 1. k-mer + revcomp (1–4) ---
    def extract_kmer_features(seq, k_max=4):
        seq = str(seq).upper().strip()
        if len(seq) == 0:
            seq = 'N' * 10  # fallback

        if DATASET_TYPE == 'rna':
            seq = seq.replace('T', 'U')
            rev_map = {'A':'U', 'U':'A', 'G':'C', 'C':'G', 'N':'N'}
            alphabet = 'AUGC'
        else:
            rev_map = {'A':'T', 'T':'A', 'G':'C', 'C':'G', 'N':'N'}
            alphabet = 'ATGC'

        # Очистка от мусора
        seq = ''.join(c for c in seq if c in alphabet)
        if len(seq) == 0:
            seq = 'N' * 10

        rev_seq = ''.join(rev_map.get(c, 'N') for c in reversed(seq))
        features = []
        for k in range(1, k_max + 1):
            all_kmers = [''.join(p) for p in itertools.product(alphabet, repeat=k)]
            count = Counter(seq[i:i+k] for i in range(max(0, len(seq)-k+1)))
            rev_count = Counter(rev_seq[i:i+k] for i in range(max(0, len(rev_seq)-k+1)))
            denom = max(len(seq) - k + 1, 1)
            for mer in all_kmers:
                features.append(count.get(mer, 0) / denom)
                features.append(rev_count.get(mer, 0) / denom)
        return np.array(features, dtype=np.float32)

    # --- 2. PseAAC (λ=3) с защитой ---
    def extract_pseaac_features(prot_seq, lam=3):
        prot_seq = str(prot_seq).upper().strip()
        aa = 'ARNDCQEGHILKMFPSTWYV'
        # Удаляем неизвестные аминокислоты
        prot_seq = ''.join(c for c in prot_seq if c in aa)
        if len(prot_seq) == 0:
            return np.zeros(20 + 20*lam, dtype=np.float32)

        L = len(prot_seq)
        count = Counter(prot_seq)
        freq = [count.get(a, 0) / L for a in aa]

        props = {
            'H': [0.62,0.29,-0.90,1.14,0.64,-0.60,0.12,-0.18,-0.37,0.10,0.31,0.11,0.42,0.23,0.16,0.13,-0.19,0.22,1.19,0.60],
            'P': [0.12,0.64,-0.78,0.23,0.10,-0.12,0.02,0.01,-0.05,0.01,0.03,0.01,0.04,0.02,0.01,0.01,-0.02,0.02,0.10,0.06],
            'V': [4.2,3.8,3.8,2.8,1.9,2.5,3.5,3.5,2.8,3.8,3.8,4.0,3.8,3.8,3.5,3.9,3.5,4.0,4.1,4.5]
        }
        corr = []
        for i in range(lam):
            for values in props.values():
                theta = 0.0
                for j in range(L - i - 1):
                    try:
                        diff = values[aa.index(prot_seq[j])] - values[aa.index(prot_seq[j+i+1])]
                        theta += diff * diff
                    except:
                        continue
                corr.append(theta / (L - i - 1) if (L - i - 1) > 0 else 0.0)
        return np.array(freq + corr, dtype=np.float32)

    # --- 3. Извлечение признаков с индексами ---
    def extract_features_with_index(df):
        X, indices, y = [], [], []
        for idx, row in df.iterrows():
            try:
                apt_seq = row['apt_seq']
                target_seq = row['target_seq']
                label = row['label']

                # Проверка типов
                if pd.isna(apt_seq) or pd.isna(target_seq):
                    continue
                if not isinstance(apt_seq, str) or not isinstance(target_seq, str):
                    continue
                if len(str(apt_seq).strip()) == 0 or len(str(target_seq).strip()) == 0:
                    continue

                apt_f = extract_kmer_features(apt_seq)
                prot_f = extract_pseaac_features(target_seq)
                X.append(np.concatenate([apt_f, prot_f]))
                indices.append(idx)
                y.append(label)
            except:
                continue
        return np.array(X), np.array(indices), np.array(y)

    # --- 4. Обучение / загрузка ---
    if os.path.exists(model_path) and os.path.exists(selector_path):
        print("Загрузка модели XGBoost и селектора...")
        model = joblib.load(model_path)
        selector = joblib.load(selector_path)
    else:
        print("Генерация признаков (train)...")
        X_train, train_indices, y_train = extract_features_with_index(train_df)
        if len(X_train) == 0:
            raise ValueError("Не удалось извлечь признаки из train_df")

        print("RF Feature Selection...")
        rf = RandomForestClassifier(n_estimators=300, max_depth=9, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        selector = SelectFromModel(rf, prefit=True)
        X_train_sel = selector.transform(X_train)
        joblib.dump(selector, selector_path)

        print("Oversampling...")
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_train_sel, y_train)

        print("Обучение XGBoost...")
        model = XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=10,
            eval_metric='logloss', random_state=42, n_jobs=-1
        )
        model.fit(X_res, y_res)
        joblib.dump(model, model_path)
        print(f"Модель сохранена: {model_path}")

    # --- 5. Инференс с ПОЛНЫМ восстановлением ---
    print("Инференс на test...")
    X_test, test_indices, y_test_valid = extract_features_with_index(test_df)

    # Если ничего не извлеклось — возвращаем нули
    if len(X_test) == 0:
        print("Нет валидных строк для предсказания.")
        n = len(test_df)
        return {
            'accuracy': 0.0,
            'report': {'0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': n}},
            'predictions': [0] * n,
            'true_labels': test_df['label'].tolist(),
            'probabilities': [0.0] * n,
            'roc_auc': 0.0
        }

    X_test_sel = selector.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # --- ВОССТАНОВЛЕНИЕ ПОЛНОЙ ДЛИНЫ ---
    n_total = len(test_df)
    full_pred = np.zeros(n_total, dtype=int)
    full_proba = np.zeros(n_total, dtype=float)
    full_true = test_df['label'].values.copy()

    full_pred[test_indices] = y_pred
    full_proba[test_indices] = y_pred_proba
    # Пропущенные строки → предсказание = 0, proba = 0.0

    report = classification_report(full_true, full_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(full_true, full_pred)

    print("--- AptaNet (XGBoost) завершён ---")
    results = {
        'accuracy': accuracy,
        'report': report,
        'predictions': full_pred.tolist(),
        'true_labels': full_true.tolist(),
        'probabilities': full_proba.tolist(),  # ← КЛЮЧЕВОЙ ПАРАМЕТР ДЛЯ БЛОКА 5
        'roc_auc': roc_auc_score(full_true, full_proba)
    }
    return results
