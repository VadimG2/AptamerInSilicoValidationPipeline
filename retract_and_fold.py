import requests
from pathlib import Path
from tqdm import tqdm
import time
import pandas as pd
import hashlib
from typing import Optional, Tuple, List
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import gzip

# Импорты путей из config
docking_dir = Path("./docking_candidates")
candidates_csv_path = docking_dir / "candidates_for_docking.csv"

# Конфигурация
API_KEY_NVIDIA = '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'  # Замените на ваш ключ

# UniProt APIs
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"
PEPTIDE_SEARCH_URL = "https://peptidesearch.uniprot.org/asyncrest/"

# AlphaFoldDB
ALPHAFOLD_ENTRY_URL = "https://alphafold.ebi.ac.uk/entry/{uniprot_id}"

# NVIDIA ESMFold
ESMFOLD_URL = "https://health.api.nvidia.com/v1/biology/nvidia/esmfold"
HEADERS_NVIDIA = {
    "Authorization": f"Bearer {API_KEY_NVIDIA}",
    "Accept": "application/json",
}

def load_sequence_from_fasta(fasta_path: Path) -> str:
    """Загружает последовательность из FASTA файла."""
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
        seq = "".join(line.strip() for line in lines if not line.startswith('>'))
    return seq

def get_sequence_hash(sequence: str) -> str:
    """Создает MD5 хэш для последовательности."""
    return hashlib.md5(sequence.encode()).hexdigest()

def create_session_with_retries() -> requests.Session:
    """Создает сессию с retry и adapter для лучшей устойчивости."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def poll_job_status(session: requests.Session, job_url: str, max_wait: int = 1200, poll_timeout: int = 60) -> Optional[str]:
    """
    Опрашивает статус задания с увеличенным timeout и логированием.
    """
    headers = {"Accept": "text/plain"}
    start_time = time.time()
    poll_count = 0
    
    while time.time() - start_time < max_wait:
        poll_count += 1
        print(f"  [PeptideSearch] Опрос {poll_count}: GET {job_url} (timeout={poll_timeout}s)")
        
        try:
            response = session.get(job_url, headers=headers, timeout=poll_timeout)
            print(f"  [PeptideSearch] Статус: {response.status_code}")
            
            if response.status_code == 303:
                retry_after = int(response.headers.get("Retry-After", 30))
                print(f"  [PeptideSearch] В процессе... Ожидание {retry_after} сек.")
                time.sleep(retry_after)
                continue
            
            elif response.status_code == 200:
                accessions = response.text.strip()
                if accessions:
                    print(f"  [PeptideSearch] Завершено! Найдено {len([a for a in accessions.split(',') if a.strip()])} кандидатов.")
                    return accessions
                else:
                    print("  [PeptideSearch] Завершено, но нет совпадений.")
                    return None
            
            elif response.status_code == 404:
                print("  [PeptideSearch] Задание не найдено (возможно, истекло).")
                return None
            
            else:
                print(f"  [PeptideSearch] Неожиданный статус {response.status_code}: {response.text}")
                time.sleep(10)
                continue
                
        except requests.exceptions.Timeout:
            print(f"  [PeptideSearch] Таймаут чтения (timeout={poll_timeout}s). Пробуем снова через 10с.")
            time.sleep(10)
            continue
        except requests.exceptions.ConnectionError as e:
            print(f"  [PeptideSearch] Ошибка соединения: {e}. Проверяем URL на HTTPS? Пробуем снова через 10с.")
            time.sleep(10)
            continue
        except Exception as e:
            print(f"  [PeptideSearch] Неожиданная ошибка: {e}. Пробуем снова через 10с.")
            time.sleep(10)
            continue
    
    print(f"  [PeptideSearch] Общий таймаут ожидания ({max_wait}s).")
    return None


# 1. Точный поиск по полному сиквенсу через Peptide Search

def search_peptide_exact(sequence: str, max_retries: int = 3) -> Optional[str]:
    """
    Ищет UniProt ID по точному совпадению полной последовательности через Peptide Search.
    Только Swiss-Prot, lEQi=on (L/I эквивалентны).
    Затем проверяет точную длину и последовательность для каждого accession.
    """
    if len(sequence) < 3:
        print("  [PeptideSearch] Последовательность слишком короткая (<3 aa). Пропуск.")
        return None
    
    payload = {
        "peps": sequence,
        "taxIds": "",
        "lEQi": "on",
        "spOnly": "off"
    }
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; version=1.1",
        "Accept": "text/plain"
    }
    
    session = create_session_with_retries()
    for attempt in range(max_retries):
        try:
            print(f"  [PeptideSearch] Попытка {attempt + 1}: Отправка POST на {PEPTIDE_SEARCH_URL} (длина {len(sequence)})...")
            post_response = session.post(PEPTIDE_SEARCH_URL, data=payload, headers=headers, timeout=180)  # Увеличен timeout
            print(f"  [PeptideSearch] POST статус: {post_response.status_code}")
            
            if post_response.status_code == 503:
                print(f"  [PeptideSearch] Сервер недоступен (503). Ожидание {10 * (2**attempt)}с...")
                time.sleep(10 * (2**attempt))
                continue
            
            if post_response.status_code != 202:
                raise requests.HTTPError(f"POST статус {post_response.status_code}: {post_response.text}")
            
            job_url = post_response.headers.get("Location")
            if not job_url:
                raise ValueError("Нет Location в заголовках ответа.")
            
            if not job_url.startswith("https://"):
                print(f"  [PeptideSearch] ВНИМАНИЕ: Location не HTTPS: {job_url}. Исправляем на HTTPS.")
                job_url = job_url.replace("http://", "https://")
            
            job_id = job_url.split("/")[-1]
            print(f"  [PeptideSearch] Задание создано: {job_id}")
            print(f"  [PeptideSearch] URL статуса: {job_url}")
            
            accessions_str = poll_job_status(session, job_url, max_wait=1200, poll_timeout=60)
            if not accessions_str:
                print(f"  [PeptideSearch] Не удалось получить результаты для {job_id}.")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None
            
            accessions = [acc.strip() for acc in accessions_str.split(",") if acc.strip()]
            print(f"  [PeptideSearch] Кандидатов: {len(accessions)}")
            
            print("  [PeptideSearch] Проверка точных матчей по последовательностям...")
            for acc in tqdm(accessions, desc="Проверка accessions", leave=False):  # Проверяем ВСЕ
                try:
                    entry_url = UNIPROT_ENTRY_URL.format(accession=acc)
                    entry_response = session.get(entry_url, timeout=30)
                    if entry_response.status_code != 200:
                        print(f"  [PeptideSearch] {acc}: HTTP {entry_response.status_code} → пропуск")
                        continue
                    
                    entry = entry_response.json()
                    uni_seq = entry.get('sequence', {}).get('value', '')
                    
                    if len(uni_seq) == len(sequence) and uni_seq == sequence:
                        entry_type = entry.get('entryType', 'Unknown')
                        print(f"  [PeptideSearch] Точный матч: {acc} (длина {len(uni_seq)}, тип: {entry_type})")
                        session.close()
                        return acc
                    else:
                        mismatch = "длина" if len(uni_seq) != len(sequence) else "последовательность"
                        print(f"  [PeptideSearch] {acc}: {mismatch} не совпадает ({len(uni_seq)} vs {len(sequence)}) → пропуск")
                
                except Exception as e:
                    print(f"  [PeptideSearch] Ошибка для {acc}: {e} → пропуск")
                    continue
            
            print("  [PeptideSearch] Точных матчей не найдено среди кандидатов.")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None
            
        except Exception as e:
            print(f"  [PeptideSearch] Попытка {attempt + 1} ошибка: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 5
                print(f"  [PeptideSearch] Ожидание {wait_time}s перед повтором.")
                time.sleep(wait_time)
            else:
                print("  [PeptideSearch] Все попытки исчерпаны.")
                return None
    
    session.close()
    return None

# 2. Получение разрешения PDB

def get_resolution(ref) -> float:
    """Безопасно извлекает разрешение PDB, обрабатывая '-' или не-числа как 99.0."""
    for p in ref.get('properties', []):
        if p['key'] == 'Resolution':
            val = p['value'].split(' ')[0]
            try:
                return float(val)
            except (ValueError, TypeError):
                print(f"  [PDB] Нечисловое разрешение '{val}' для {ref['id']} → трактуем как 99.0")
                return 99.0
    return 99.0

# 2.5 Скачивание AlphaFoldDB модели

def download_alphafold_model(uniprot_id: str, save_path: Path, session: requests.Session) -> Optional[str]:
    """
    Парсит страницу https://alphafold.ebi.ac.uk/entry/{uniprot_id}
    Извлекает реальную ссылку на PDB файл (F1-model_vX.pdb[.gz])
    Скачивает и (если .gz) распаковывает.
    Возвращает версию модели, если успешно, иначе None.
    """
    entry_url = ALPHAFOLD_ENTRY_URL.format(uniprot_id=uniprot_id)
    try:
        print(f"  [AlphaFoldDB] Парсинг страницы: {entry_url}")
        response = session.get(entry_url, timeout=60)
        if response.status_code != 200:
            print(f"  [AlphaFoldDB] Не удалось загрузить страницу entry (HTTP {response.status_code})")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        # Ищем ссылки на PDB: href содержит 'AF-{uniprot_id}-F1-model_v' и '.pdb'
        pdb_links = soup.find_all('a', href=lambda x: x and 'AF-{}-F1-model_v'.format(uniprot_id) in x and '.pdb' in x)
        
        if not pdb_links:
            print(f"  [AlphaFoldDB] Ссылки на PDB не найдены на странице для {uniprot_id}")
            return None

        # Берем первую (обычно F1, vX.pdb или .gz)
        pdb_href = pdb_links[0]['href']
        pdb_url = "https://alphafold.ebi.ac.uk" + pdb_href if pdb_href.startswith('/') else pdb_href
        print(f"  [AlphaFoldDB] Найдена ссылка: {pdb_url}")
        
        # Скачиваем
        download_response = session.get(pdb_url, timeout=120)
        if download_response.status_code != 200:
            print(f"  [AlphaFoldDB] Не удалось скачать PDB (HTTP {download_response.status_code})")
            return None

        # Если .gz — распаковываем
        content = download_response.content
        if pdb_url.endswith('.gz'):
            print(f"  [AlphaFoldDB] Распаковка .gz...")
            content = gzip.decompress(content)
            version = pdb_url.split('model_v')[1].split('.pdb')[0]
        else:
            version = pdb_url.split('model_v')[1].split('.pdb')[0]

        # Проверяем PDB
        if b"ATOM" not in content:
            print(f"  [AlphaFoldDB] Файл не содержит ATOM записей")
            return None

        with open(save_path, 'wb') as f:
            f.write(content)
        with open(save_path, 'a') as f:
            f.write(f"\nREMARK   SOURCE: AlphaFoldDB {uniprot_id} (v{version})\n".encode())

        print(f"  [AlphaFoldDB] Сохранено: {save_path} (v{version})")
        return version
        
    except Exception as e:
        print(f"  [AlphaFoldDB] Ошибка для {uniprot_id}: {e}")
        return None


# 3. Получение структуры (PDB или AlphaFold) по UniProt ID

def get_structure_from_uniprot(uniprot_id: str, target_seq_len: int, session: requests.Session) -> Optional[Tuple[str, str, str]]:
    """
    Возвращает (struct_id, sequence, source_type)
    source_type: 'pdb', 'alphafold', None
    """
    try:
        entry_url = UNIPROT_ENTRY_URL.format(accession=uniprot_id)
        print(f"  [UniProt] Запрос: {entry_url}")
        response = session.get(entry_url, timeout=30)
        response.raise_for_status()
        entry = response.json()

        uni_seq = entry.get('sequence', {}).get('value', '')
        print(f"  [UniProt] Длина последовательности: {len(uni_seq)}")
        if abs(len(uni_seq) - target_seq_len) > 0:
            print(f"  [UniProt] Длина не совпадает: {len(uni_seq)} ≠ {target_seq_len} → пропуск")
            return None

        # --- 1. Ищем PDB ---
        pdb_refs = [r for r in entry.get('uniProtKBCrossReferences', []) if r['database'] == 'PDB']
        print(f"  [PDB] Найдено PDB ссылок: {len(pdb_refs)}")
        if pdb_refs:
            pdb_refs.sort(key=get_resolution)
            ref = pdb_refs[0]
            pdb_id = ref['id'].lower()
            resolution = next((p['value'] for p in ref.get('properties', []) if p['key'] == 'Resolution'), 'N/A')
            print(f"  [PDB] Лучший: {pdb_id} (разрешение: {resolution})")
            return pdb_id, uni_seq, 'pdb'

        # Если PDB нет — AlphaFoldDB
        print(f"  [AlphaFoldDB] PDB не найден. Проверка страницы entry для {uniprot_id}...")
        return uniprot_id, uni_seq, 'alphafold'

    except Exception as e:
        print(f"  [UniProt] Ошибка для {uniprot_id}: {e}")
        return None

# 4. Скачивание PDB

def download_pdb(pdb_id: str, save_path: Path, session: requests.Session) -> bool:
    """Скачивает PDB с использованием сессии."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        print(f"  [Download] Скачивание {pdb_id} с {url}")
        response = session.get(url, timeout=30)
        response.raise_for_status()
        with open(save_path, 'w') as f:
            f.write(response.text)
        print(f"  [Download] Сохранено: {save_path}")
        return True
    except Exception as e:
        print(f"  [Download] Ошибка для {pdb_id}: {e}")
        return False


if not candidates_csv_path.exists():
    print(f"Ошибка: Файл {candidates_csv_path} не найден.")
    raise SystemExit

candidates_df = pd.read_csv(candidates_csv_path)
print(f"Загружено {len(candidates_df)} кандидатов.")

unique_sequences = {}
seq_to_candidates = {}

print("Сбор уникальных последовательностей...")
for _, row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="Сбор сиквенсов"):
    candidate_id = row['candidate_id']
    fasta_path = docking_dir / candidate_id / "protein.fasta"

    if not fasta_path.exists():
        print(f"  [Предупреждение] {fasta_path} не найден → пропуск")
        continue

    seq = load_sequence_from_fasta(fasta_path)
    if not seq:
        continue

    seq_hash = get_sequence_hash(seq)
    if seq_hash not in unique_sequences:
        unique_sequences[seq_hash] = {
            'sequence': seq,
            'length': len(seq),
            'source': None,
            'uniprot_id': None,
            'pdb_id': None
        }
    seq_to_candidates.setdefault(seq_hash, []).append(candidate_id)

print(f"Найдено {len(unique_sequences)} уникальных последовательностей.")

# Глобальная сессия для всех запросов
global_session = create_session_with_retries()

for seq_hash, info in tqdm(unique_sequences.items(), desc="Поиск структур"):
    seq = info['sequence']
    candidate_ids = seq_to_candidates[seq_hash]
    short_hash = seq_hash[:8]

    print(f"\n[Поиск] Сиквенс {short_hash} (длина {len(seq)})")

    # Этап 1: Точный поиск в Peptide Search
    uniprot_id = search_peptide_exact(seq)
    if uniprot_id:
        unique_sequences[seq_hash]['uniprot_id'] = uniprot_id

        # Этап 2: Получаем структуру
        structure_info = get_structure_from_uniprot(uniprot_id, len(seq), global_session)
        if structure_info:
            struct_id, uni_seq, source_type = structure_info
            unique_sequences[seq_hash]['source'] = source_type

            success = False
            for cid in candidate_ids:
                candidate_dir = docking_dir / cid
                candidate_dir.mkdir(parents=True, exist_ok=True)
                pdb_path = candidate_dir / "protein_3D.pdb"

                if source_type == 'pdb':
                    unique_sequences[seq_hash]['pdb_id'] = struct_id
                    if download_pdb(struct_id, pdb_path, global_session):
                        with open(pdb_path, 'a') as f:
                            f.write(f"\nREMARK   SOURCE: UniProt {uniprot_id}, PDB {struct_id}\n")
                        success = True
                    print(f"  [УСПЕХ] PDB сохранён: {struct_id}")

                elif source_type == 'alphafold':
                    downloaded_version = download_alphafold_model(uniprot_id, pdb_path, global_session)
                    if downloaded_version:
                        unique_sequences[seq_hash]['pdb_id'] = f"AF-{uniprot_id}-F1-v{downloaded_version}"
                        print(f"  [УСПЕХ] AlphaFold модель сохранена: AF-{uniprot_id}-F1-v{downloaded_version}")
                        success = True
                    else:
                        print("  [AlphaFoldDB] Не удалось скачать модель. Переход к ESMFold.")

            if success:
                continue
        else:
            print("  [UniProt] Не удалось получить структуру. Переход к ESMFold.")
    else:
        print("  [PeptideSearch] Точный матч не найден. Переход к ESMFold.")

    # Этап 3: ESMFold
    print("  [ESMFold] Предсказание...")
    payload = {"sequence": seq}
    try:
        response = global_session.post(ESMFOLD_URL, headers=HEADERS_NVIDIA, json=payload, timeout=600)
        response.raise_for_status()

        pdb_content = ""
        try:
            json_response = response.json()
            pdb_list = json_response.get("pdbs", [])
            if pdb_list:
                pdb_content = pdb_list[0]
        except json.JSONDecodeError:
            pdb_content = response.text

        if "ATOM" in pdb_content:
            for cid in candidate_ids:
                candidate_dir = docking_dir / cid
                candidate_dir.mkdir(parents=True, exist_ok=True)
                pdb_path = candidate_dir / "protein_3D.pdb"
                with open(pdb_path, "w") as f:
                    f.write(pdb_content)
                    f.write(f"\nREMARK   SOURCE: ESMFold (predicted)\n")
            unique_sequences[seq_hash]['source'] = 'esmfold'
            print("  [УСПЕХ] ESMFold: структура предсказана")
        else:
            raise ValueError("Невалидный PDB от ESMFold")
    except Exception as e:
        error_msg = str(e)
        print(f"  [ОШИБКА] ESMFold: {error_msg}")
        for cid in candidate_ids:
            err_path = docking_dir / cid / "PROTEIN_ERROR.txt"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with open(err_path, "w") as f:
                f.write(f"ESMFold ошибка: {error_msg}\n")
        unique_sequences[seq_hash]['source'] = 'failed'
    time.sleep(1)  # Пауза между ESMFold

global_session.close()

print("\n" + "="*80)
print("ИТОГОВАЯ СТАТИСТИКА:")
stats = pd.Series([info.get('source', 'unknown') for info in unique_sequences.values()]).value_counts()

print(f"  UniProt + PDB:         {stats.get('pdb', 0)}")
print(f"  UniProt + AlphaFoldDB: {stats.get('alphafold', 0)}")
print(f"  ESMFold (предсказано): {stats.get('esmfold', 0)}")
print(f"  Ошибки:                {stats.get('failed', 0)}")
print(f"  --------------------------------------------------")
print(f"  Всего уникальных:      {len(unique_sequences)}")
print("="*80)
print("Поиск 3D-структур завершён.")

with open("block2.done", "w") as f:
    f.write("Done\n")
print("block2.done создан — блок 2 завершён.")
