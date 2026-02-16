# Automated Validation Pipeline for Aptamer Generation Results

## Pipeline Overview

The entire workflow is divided into six sequential blocks:

### Block 1: ML Screening and Consensus
This block performs initial processing of the input dataset (`Aptamers_and_targets.csv`).  
**Actions:**
- Data preprocessing and train/test splitting (`preprocess_and_split.py`).
- Running several neural network models (**Apipred**, **AptaTrans**, **AptaNet**) to predict binding probability (`apipred_benchmark.py`, `aptatrans_benchmark.py`, `aptanet_benchmark.py`).
- Forming a consensus decision based on the models’ outputs (`run_benchmark_and_consensus.py`).
- Selecting the best candidates for subsequent structural modeling (`run_benchmark_and_consensus.py`).  
**Result:** `candidates_for_docking.csv` and a prepared folder structure for each candidate.

### Block 2: Protein Sequence Processing
For each protein target, its 3D structure is obtained either by folding from the FASTA sequence using ESMFold (primary option) or, if ESMFold fails (e.g., due to size or ligand issues), by searching the AlphaFold database.  
**Actions:** Protein folding and sequence validation (`retract_and_fold.py`).  
**Result:** Protein 3D structures (`protein_3D.pdb`) for each target.

### Block 3: Secondary Structure Prediction
Prediction of the aptamer’s secondary structure (dot‑bracket notation) to accelerate subsequent 3D folding.  
**Tool:** RNAfold (or equivalent algorithms in `2D_aptamer_prediction.py`).  
**Result:** `aptamer.ss` files for each candidate.

### Block 4: 3D Aptamer Modeling
Generation of the aptamer’s tertiary structure based on its secondary structure.  
- **Step 4.1 (NSP Assembly):** Production of a pool of coarse 3D models using **NSP** (NSP/cirRNA_and_DNA/nsp).  
- **Step 4.2 (Refinement):** Geometric refinement of each model.  
- **Step 4.3 (Selection):** Selection of the best model based on energy or structural criteria and cleanup of temporary files.  
**Result:** `aptamer_3D.pdb` ready for docking.

### Block 5: Molecular Docking
Modeling of the protein–aptamer interaction.  
- **Docking:** **HDOCK** is used to predict the complex between `protein_3D.pdb` and `aptamer_3D.pdb`.  
- **Aggregation:** Collection of scoring results for all candidates into a single CSV file.  
- **Validation:** Automatic check of the generated PDB complex files before passing them to MD:
  - Presence of atoms.
  - Minimum atom count thresholds (protein > 50, RNA > 20).
  - Generation of a `.validated` flag only for correct structures.  
**Result:** `final_ranked_candidates_hdock.csv` (ranks and scores) and validated PDB complexes.

### Block 6: Coarse‑Grained Molecular Dynamics
Final stability assessment of the complexes using the **Martini 3** force field and **GROMACS**. Only complexes that passed validation in Block 5 proceed.  
- **Step 6.1 (Prep):** Preparation of all‑atom structures, chain splitting, and PDB header fixing.  
- **Step 6.2 (Martinize):** Conversion to coarse‑grained representation:
  - Protein: `martinize2`.
  - RNA: `reForge` (script `martinize_rna_v3.0.0.py`).
  - Merging of topology files (`.itp`).  
- **Step 6.3 (Build System):** System assembly, cubic box definition, solvation (Martini water), and ion addition for charge neutralization.  
- **Step 6.4 (Minimization):** Two‑step energy minimization (Steepest Descent).  
- **Step 6.5 (Equilibration):** System equilibration in NVT and NPT ensembles.  
- **Step 6.6 (Production):** Production MD run on GPU.  
**Result:** Trajectory (`md.xtc`) and final structure (`md.gro`) for stability analysis.

---

## System Requirements
*(To be added: required software packages and automatic environment setup per step will be integrated soon.)*
