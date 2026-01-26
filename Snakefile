import warnings
import csv
import os
from pathlib import Path
from Bio.PDB import PDBParser
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
os.makedirs("logs", exist_ok=True)

ROOT = os.path.abspath(os.path.dirname(workflow.snakefile))
MDP_DIR = "mdp"
configfile: "config.yaml"

def find_models(path_to_candidates):
    """
    Сканирует директорию с кандидатами и возвращает список кортежей (candidate, model).
    """
    models_list = []
    candidate_dirs = Path(path_to_candidates).glob("candidate_*")
    for cand_dir in candidate_dirs:
        if not cand_dir.is_dir():
            continue
        candidate_name = cand_dir.name
        pdb_files = [p for p in cand_dir.glob("*.pdb") if "protein_3D" not in p.name]
        for pdb_path in pdb_files:
            model_name = pdb_path.stem
            models_list.append((candidate_name, model_name))
    return models_list

MODELS = find_models("docking_candidates")

# ГЛОБАЛЬНЫЕ ВЫХОДНЫЕ ФАЙЛЫ
DOCKING_SCORE_FILES = expand("docking_results/{candidate}/{model}.score.txt",
                             zip,
                             candidate=[m[0] for m in MODELS],
                             model=[m[1] for m in MODELS])

DOCKING_PDB_FILES = expand("docking_results/{candidate}/{model}.top100.pdb",
                           zip,
                           candidate=[m[0] for m in MODELS],
                           model=[m[1] for m in MODELS])

rule all:
    input:
        "docking_results/final_ranked_candidates_hdock.csv",
        "block6.done"

# Блок 1: Обработка датасета, бенчмарки моделей, консенсус, создание папок candidate_n
rule block1:
    input:
        "Aptamers_and_targets.csv"
    output:
        csv="docking_candidates/candidates_for_docking.csv",
        done=touch("block1.done")
    log:
        "logs/block1.log"
    shell:
        """
        (
        python preprocess_and_split.py &&
        python apipred_benchmark.py &&
        python aptatrans_benchmark.py &&
        python aptanet_benchmark.py &&
        python run_benchmarks_and_consensus.py
        ) > {log} 2>&1
        """

# Блок 2: Оценка уникальных последовательностей
rule block2:
    input:
        "block1.done"
    output:
        done=touch("block2.done")
    log:
        "logs/block2.log"
    shell:
        "python retract_and_fold.py > {log} 2>&1"

# Блок 3: Предсказание вторичной структуры аптамера (RNAFold)
# Этот блок должен создать файлы aptamer.ss для каждого кандидата

rule block3:
    input:
        "block2.done"
    output:
        # Явно объявляем, что этот блок создает все нужные .ss файлы
        ss_files=expand("docking_candidates/{candidate_id}/aptamer.ss", candidate_id=CANDIDATE_IDS),
        done=touch("block3.done")
    log:
        "logs/block3.log"
    shell:
        "python 2D_aptamer_prediction.py > {log} 2>&1"

# Блок 4: 3D структура аптамера
# Шаг 4.1: Запуск NSP для генерации 3D моделей по 2D структуре

rule nsp_predict_3d:
    input:
        ss_file="docking_candidates/{candidate_id}/aptamer.ss"
    output:
        touch(expand("docking_candidates/{{candidate_id}}/{candidate_id}.pred{i}.pdb", i=range(1, NUM_MODELS + 1)))
    params:
        nsp_exec="NSP/cirRNA_and_DNA/nsp",
        out_dir="docking_candidates/{candidate_id}",
        prefix="{candidate_id}"
    log:
        "logs/nsp/{candidate_id}.log"
    shell:
        """
        SEQ=$(sed -n '2p' {input.ss_file})
        SS=$(sed -n '3p' {input.ss_file} | awk '{{print $1}}')

        # Проверка, что во вторичной структуре есть пары
        if [[ "$SS" != *"("* && "$SS" != *")"* ]]; then
            echo "Вторичная структура для {wildcards.candidate_id} не содержит пар. Пропускаем."
            touch {output}
            exit 0
        fi

        (cd {params.out_dir} && \
        ../../{params.nsp_exec} assemble \
            -name {params.prefix} \
            -seq "$SEQ" \
            -ss "$SS" \
            -n {NUM_MODELS}) > {log} 2>&1
        """

# Шаг 4.2: Уточнение каждой предсказанной 3D модели

rule refine_3d_model:
    input:
        pdb="docking_candidates/{candidate_id}/{candidate_id}.pred{model_num}.pdb",
        ss_file="docking_candidates/{candidate_id}/aptamer.ss",
        script="refine_md.py"
    output:
        "docking_candidates/{candidate_id}/refined_md_{candidate_id}.pred{model_num}.pdb"
    log:
        "logs/refine_md/{candidate_id}_{model_num}.log"
    shell:
        "python {input.script} --pdb {input.pdb} --ss_file {input.ss_file} --output {output} > {log} 2>&1"

# Шаг 4.3: Выбор лучшей модели и очистка

rule select_best_3d_structure:
    input:
        refined_pdbs=expand("docking_candidates/{{candidate_id}}/refined_md_{{candidate_id}}.pred{model_num}.pdb", model_num=range(1, NUM_MODELS + 1))
    output:
        final_pdb="docking_candidates/{candidate_id}/aptamer_3D.pdb"
    shell:
        """
        BEST_PDB=$(ls -S {input.refined_pdbs} | head -n 1)
        if [ -n "$BEST_PDB" ]; then
            cp "$BEST_PDB" {output.final_pdb}

            # Очистка временных файлов
            rm -f docking_candidates/{wildcards.candidate_id}/*.pred*.pdb
            rm -f {input.refined_pdbs}
        else
            echo "Не найдено уточненных PDB файлов для {wildcards.candidate_id}" >&2
            exit 1
        fi
        """

# Шаг 4.4: Правило-агрегатор для всего Блока 4
rule block4:
    input:
        expand("docking_candidates/{candidate_id}/aptamer_3D.pdb", candidate_id=CANDIDATE_IDS)
    output:
        touch("block4.done")


# Блок 5: HDOCK

rule block5_dock_individual:
    input:
        receptor = "docking_candidates/{candidate}/protein_3D.pdb",
        ligand = "docking_candidates/{candidate}/{model}.pdb"
    output:
        score_file = "docking_results/{candidate}/{model}.score.txt",
        complex_pdb = "docking_results/{candidate}/{model}.top100.pdb"
    params:
        hdock_path = (Path(config["hdock_dir"]) / "hdock").resolve(),
        createpl_path = (Path(config["hdock_dir"]) / "createpl").resolve(),
        nmax = config["createpl_nmax"],
        candidate = "{candidate}",
        model = "{model}"
    log:
        "logs/docking/{candidate}/{model}.log"
    shell:
        """
        set -e
        echo "=== Докинг {params.candidate}/{params.model} ===" > {log}

        python scripts/run_individual_docking.py \
            {input.receptor} {input.ligand} \
            {output.score_file} {output.complex_pdb} \
            {params.hdock_path} {params.createpl_path} {params.nmax} \
            {params.candidate} {params.model} >> {log} 2>&1

        if [ ! -s {output.complex_pdb} ]; then
            echo "WARNING: Пустой результат докинга" >> {log}
        elif ! grep -q "^ATOM" {output.complex_pdb}; then
            echo "WARNING: Нет ATOM записей в результате" >> {log}
        else
            echo "SUCCESS: Докинг завершён ($(grep -c "^MODEL" {output.complex_pdb} || echo 1) моделей)" >> {log}
        fi
        """

checkpoint block5_aggregate_results:
    input:
        DOCKING_SCORE_FILES
    output:
        final_csv = "docking_results/final_ranked_candidates_hdock.csv"
    shell:
        "python scripts/aggregate_docking_results.py {input} {output}"

# Блок 5.5: Валидация результатов докинга

rule block5_validate_individual:
    input:
        pdb = "docking_results/{candidate}/{model}.top100.pdb",
        score = "docking_results/{candidate}/{model}.score.txt",
        csv = "docking_results/final_ranked_candidates_hdock.csv"
    output:
        flag = "docking_results/{candidate}/{model}.validated"
    params:
        candidate = "{candidate}",
        model = "{model}"
    log:
        "logs/validation/{candidate}/{model}.log"
    run:
        import os
        from pathlib import Path

        log_path = str(log)
        pdb_path = str(input.pdb)
        candidate = params.candidate
        model = params.model

        with open(log_path, 'w') as log_f:
            log_f.write(f"=== Валидация {candidate}/{model} ===\n")

            # 1. Проверка существования и размера файла
            if not os.path.exists(pdb_path):
                log_f.write(f"FAIL: PDB файл не существует\n")
                raise ValueError(f"Missing PDB: {pdb_path}")

            pdb_size = os.path.getsize(pdb_path)
            if pdb_size < 1024:
                log_f.write(f"FAIL: PDB файл слишком мал ({pdb_size} байт)\n")
                raise ValueError(f"PDB too small: {pdb_size} bytes")

            log_f.write(f"Размер файла: {pdb_size} байт\n")

            # 2. Проверка наличия ATOM записей
            has_atoms = False
            protein_residues = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
                              'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','CYX'}
            rna_residues = {'A','G','C','U','DA','DG','DC','DT','RA','RG','RC','RU'}

            protein_count = 0
            rna_count = 0
            model_count = 0

            try:
                with open(pdb_path, 'r') as f:
                    for line in f:
                        if line.startswith("MODEL"):
                            model_count += 1
                        elif line.startswith("ATOM") or line.startswith("HETATM"):
                            has_atoms = True
                            res_name = line[17:20].strip()
                            if res_name in protein_residues:
                                protein_count += 1
                            elif res_name in rna_residues:
                                rna_count += 1
            except Exception as e:
                log_f.write(f"FAIL: Ошибка чтения PDB: {e}\n")
                raise

            if not has_atoms:
                log_f.write("FAIL: Нет ATOM/HETATM записей\n")
                raise ValueError("No ATOM records found")

            log_f.write(f"Найдено моделей: {model_count if model_count > 0 else 1}\n")
            log_f.write(f"Белковых атомов: {protein_count}\n")
            log_f.write(f"РНК атомов: {rna_count}\n")

            # 3. Проверка минимального количества атомов для комплекса
            MIN_PROTEIN_ATOMS = 50
            MIN_RNA_ATOMS = 20

            if protein_count < MIN_PROTEIN_ATOMS:
                log_f.write(f"FAIL: Недостаточно белковых атомов ({protein_count} < {MIN_PROTEIN_ATOMS})\n")
                raise ValueError(f"Insufficient protein atoms: {protein_count}")

            if rna_count < MIN_RNA_ATOMS:
                log_f.write(f"FAIL: Недостаточно РНК атомов ({rna_count} < {MIN_RNA_ATOMS})\n")
                raise ValueError(f"Insufficient RNA atoms: {rna_count}")

            # 4. Все проверки пройдены
            log_f.write(f"\nVALIDATION PASSED\n")
            log_f.write(f"Комплекс может быть использован для MD симуляции\n")

        # Создаём флаг валидации
        with open(str(output.flag), 'w') as f:
            f.write(f"VALIDATED: {candidate}/{model}\n")
            f.write(f"Protein atoms: {protein_count}\n")
            f.write(f"RNA atoms: {rna_count}\n")
            f.write(f"Models: {model_count if model_count > 0 else 1}\n")

# Блок 6: CGMD - валидация при помощи крупнозернистой MD

def get_validated_complexes(wildcards):
    checkpoint_output = checkpoints.block5_aggregate_results.get(**wildcards).output.final_csv

    if not os.path.exists(checkpoint_output):
        print("[MD] CSV не найден")
        return []

    validated = []

    with open(checkpoint_output) as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if len(row) < 2:
                continue

            c, m = row[0].strip(), row[1].strip()

            # Проверяем существование PDB
            pdb_path = Path(f"docking_results/{c}/{m}.top100.pdb")
            if not pdb_path.exists():
                print(f"[MD] Пропускаем {c}/{m}: PDB не существует")
                continue

            # Проверяем размер PDB
            if pdb_path.stat().st_size < 1024:
                print(f"[MD] Пропускаем {c}/{m}: PDB слишком мал")
                continue

            # Проверяем наличие валидации
            flag_path = Path(f"docking_results/{c}/{m}.validated")
            if not flag_path.exists():
                print(f"[MD] Пропускаем {c}/{m}: не прошёл валидацию")
                continue

            validated.append((c, m))

    print(f"\n[MD] Найдено {len(validated)} валидных комплексов для симуляции\n")

    return expand("md_runs/{candidate}/{model}/md.xtc",
                  zip,
                  candidate=[x[0] for x in validated],
                  model=[x[1] for x in validated])

rule block6:
    input: get_validated_complexes
    output: touch("block6.done")

# 6.1. Подготовка структуры

rule md_prep_structure:
    input:
        pdb = "docking_results/{candidate}/{model}.top100.pdb",
        flag = "docking_results/{candidate}/{model}.validated",
        script = "scripts/prepare_structure.py"
    output:
        prot_fixed = "md_runs/{candidate}/{model}/protein_full_atom_fixed.pdb",
        rna_full = "md_runs/{candidate}/{model}/rna_full_atom.pdb",
        ss_txt = "md_runs/{candidate}/{model}/ss.txt"
    params:
        work_dir = "md_runs/{candidate}/{model}"
    log:
        "logs/md_prep/{candidate}/{model}.log"
    shell:
        """
        mkdir -p {params.work_dir}

        echo "Подготовка структуры {wildcards.candidate}/{wildcards.model}" > {log}

        python {input.script} {input.pdb} {output.prot_fixed} {output.rna_full} {output.ss_txt} >> {log} 2>&1

        # Валидация выходов
        if [ ! -s {output.prot_fixed} ] || [ ! -s {output.rna_full} ]; then
            echo "FAIL: prepare_structure создал пустые файлы" >> {log}
            exit 1
        fi

        echo "Структура подготовлена успешно" >> {log}
        """

# 6.2. Мартинизация

rule md_martinize:
    input:
        prot_fixed = "md_runs/{candidate}/{model}/protein_full_atom_fixed.pdb",
        rna_raw = "md_runs/{candidate}/{model}/rna_full_atom.pdb",
        ss_txt = "md_runs/{candidate}/{model}/ss.txt",
        merge_script = "scripts/merge_itp.py"
    output:
        prot_cg = "md_runs/{candidate}/{model}/protein_cg.pdb",
        prot_itp = "md_runs/{candidate}/{model}/protein.itp",
        rna_cg = "md_runs/{candidate}/{model}/rna_cg.pdb",
        rna_itp = "md_runs/{candidate}/{model}/rna.itp"
    params:
        work_dir = "md_runs/{candidate}/{model}",
        reforge_script = "../../../reForge/scripts/martinize_rna_v3.0.0.py",
        merge_script_rel = "../../../scripts/merge_itp.py"
    log:
        "logs/md_martinize/{candidate}/{model}.log"
    shell:
        """
        cd {params.work_dir}

        echo "Мартинизация" > ../../../{log}

        martinize2 -f protein_full_atom_fixed.pdb -x protein_cg.pdb \
            -ff martini3001 -ss "$(cat ss.txt)" -o molecule_0.itp >> ../../../{log} 2>&1

        python {params.merge_script_rel} molecule_0.itp protein.itp
        rm molecule_0.itp

        python {params.reforge_script} -f rna_full_atom.pdb \
            -os rna_cg.pdb -ot rna.itp -elastic no >> ../../../{log} 2>&1

        echo "Мартинизация завершена успешно" >> ../../../{log}
        """

# 6.3. Сборка системы

rule md_build_system:
    input:
        prot_cg = "md_runs/{candidate}/{model}/protein_cg.pdb",
        rna_cg = "md_runs/{candidate}/{model}/rna_cg.pdb",
        prot_itp = "md_runs/{candidate}/{model}/protein.itp",
        rna_itp = "md_runs/{candidate}/{model}/rna.itp"
    output:
        gro = "md_runs/{candidate}/{model}/system_final.gro",
        top = "md_runs/{candidate}/{model}/topol.top",
        ndx = "md_runs/{candidate}/{model}/index.ndx"
    params:
        work_dir = "md_runs/{candidate}/{model}",
        martini_itp = "../../../resources/martini_v300/martini_v3.0.0.itp",
        martini_rna = "../../../reForge/scripts/martinize_rna_v3.0.0_itps/martini_v3.0.0_rna.itp",
        martini_nucl = "../../../resources/martini_v300/martini_v3.0.0_nucleobases_v1.itp",
        martini_solv = "../../../resources/martini_v300/martini_v3.0.0_solvents_v1.itp",
        martini_ions = "../../../resources/martini_v300/martini_v3.0.0_ions_v1.itp",
        water_gro = "../../../resources/reForge/reforge/martini/datdir/water.gro",
        minim_mdp = "../../../mdp/minim.mdp"
    log:
        "logs/md_build/{candidate}/{model}.log"
    shell:
        """
        cd {params.work_dir}

        echo "Сборка системы" > ../../../{log}

        cat protein_cg.pdb > complex_cg_final.pdb
        echo "TER" >> complex_cg_final.pdb
        cat rna_cg.pdb >> complex_cg_final.pdb

        gmx editconf -f complex_cg_final.pdb -o complex_cg_final.gro >> ../../../{log} 2>&1

        cat << EOF > topol.top
#include "{params.martini_itp}"
#include "{params.martini_rna}"
#include "{params.martini_nucl}"
#include "{params.martini_solv}"
#include "{params.martini_ions}"

#include "protein.itp"
#include "rna.itp"

[ system ]
Protein-RNA Complex

[ molecules ]
protein            1
molecule           1
EOF

        gmx editconf -f complex_cg_final.gro -o system_boxed.gro -c -d 1.2 -bt cubic >> ../../../{log} 2>&1
        gmx solvate -cp system_boxed.gro -cs {params.water_gro} -o system_solvated.gro -p topol.top >> ../../../{log} 2>&1
        gmx grompp -f {params.minim_mdp} -c system_solvated.gro -p topol.top -o ions.tpr -maxwarn 50 >> ../../../{log} 2>&1
        echo "W" | gmx genion -s ions.tpr -o system_final.gro -p topol.top -pname NA -nname CL -neutral >> ../../../{log} 2>&1
        echo -e "1|2\nname 18 Solute\n!18\nname 19 Solvent\nq" | gmx make_ndx -f system_final.gro -o index.ndx >> ../../../{log} 2>&1

        echo "Система собрана успешно" >> ../../../{log}
        """

# 6.4. Минимизация

rule md_minimization:
    input:
        gro = "md_runs/{candidate}/{model}/system_final.gro",
        top = "md_runs/{candidate}/{model}/topol.top",
        ndx = "md_runs/{candidate}/{model}/index.ndx"
    output:
        em2_gro = "md_runs/{candidate}/{model}/em2.gro",
        em2_tpr = "md_runs/{candidate}/{model}/em2.tpr"
    params:
        work_dir = "md_runs/{candidate}/{model}",
        minim_mdp = "../../../mdp/minim.mdp",
        minim2_mdp = "../../../mdp/minim2.mdp"
    threads: 16
    log:
        "logs/md_minim/{candidate}/{model}.log"
    shell:
        """
        cd {params.work_dir}

        echo "Минимизация..." > ../../../{log}

        gmx grompp -f {params.minim_mdp} -c system_final.gro -r system_final.gro \
            -p topol.top -o em.tpr -maxwarn 50 >> ../../../{log} 2>&1
        gmx mdrun -v -deffnm em -ntomp {threads} >> ../../../{log} 2>&1

        gmx grompp -f {params.minim2_mdp} -c em.gro -r em.gro \
            -p topol.top -o em2.tpr -maxwarn 50 >> ../../../{log} 2>&1
        gmx mdrun -v -deffnm em2 -ntomp {threads} >> ../../../{log} 2>&1

        echo "Минимизация завершена успешно" >> ../../../{log}
        """


# 6.5. Уравновешивание

rule md_equilibration:
    input:
        gro = "md_runs/{candidate}/{model}/em2.gro",
        top = "md_runs/{candidate}/{model}/topol.top",
        ndx = "md_runs/{candidate}/{model}/index.ndx"
    output:
        npt_gro = "md_runs/{candidate}/{model}/npt.gro",
        npt_cpt = "md_runs/{candidate}/{model}/npt.cpt"
    params:
        work_dir = "md_runs/{candidate}/{model}",
        nvt_mdp = "../../../mdp/nvt.mdp",
        npt_mdp = "../../../mdp/npt.mdp"
    resources:
        gpu = 1
    threads: 8
    log:
        "logs/md_equil/{candidate}/{model}.log"
    shell:
        """
        cd {params.work_dir}

        echo "Уравновешивание..." > ../../../{log}

        gmx grompp -f {params.nvt_mdp} -c em2.gro -r em2.gro -p topol.top \
            -n index.ndx -o nvt.tpr -maxwarn 50 >> ../../../{log} 2>&1
        gmx mdrun -v -deffnm nvt -ntomp {threads} \
            -nb gpu -bonded gpu -pme gpu -update gpu >> ../../../{log} 2>&1

        gmx grompp -f {params.npt_mdp} -c nvt.gro -t nvt.cpt -p topol.top \
            -n index.ndx -o npt.tpr -maxwarn 50 >> ../../../{log} 2>&1
        gmx mdrun -v -deffnm npt -ntomp {threads} \
            -nb gpu -bonded gpu -pme gpu -update gpu >> ../../../{log} 2>&1

        echo "Уравновешивание завершено успешно" >> ../../../{log}
        """

# 6.6. Продакшн

rule md_production:
    input:
        gro = "md_runs/{candidate}/{model}/npt.gro",
        cpt = "md_runs/{candidate}/{model}/npt.cpt",
        top = "md_runs/{candidate}/{model}/topol.top",
        ndx = "md_runs/{candidate}/{model}/index.ndx"
    output:
        xtc = "md_runs/{candidate}/{model}/md.xtc",
        gro = "md_runs/{candidate}/{model}/md.gro"
    params:
        work_dir = "md_runs/{candidate}/{model}",
        md_mdp = "../../../mdp/md.mdp"
    resources:
        gpu = 1
    threads: 8
    log:
        "logs/md_prod/{candidate}/{model}.log"
    shell:
        """
        cd {params.work_dir}

        echo "Продуктивная симуляция..." > ../../../{log}

        gmx grompp -f {params.md_mdp} -c npt.gro -t npt.cpt -p topol.top \
            -n index.ndx -o md.tpr -maxwarn 50 >> ../../../{log} 2>&1
        gmx mdrun -deffnm md -nb gpu -bonded gpu -pme gpu -update gpu \
            -ntmpi 1 -ntomp {threads} >> ../../../{log} 2>&1

        echo "Продуктивная MD завершена" >> ../../../{log}
        """
