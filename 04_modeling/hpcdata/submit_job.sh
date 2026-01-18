#!/bin/bash -l
#SBATCH --job-name=unet_cityscapes_giga_run
#SBATCH --output=logs/output_%j.txt     # Log-Dateien (werden unten zusätzlich in Run-Ordner umgeleitet)
#SBATCH --error=logs/error_%j.txt

# --- Hardware Ressourcen (Alex Cluster) ---
#SBATCH --gres=gpu:a100:1 -C a100_80     # 1 GPU a100 80GB
#SBATCH --nodes=1                               # 1 Knoten
#SBATCH --ntasks=1                              # 1 Prozess
#SBATCH --cpus-per-task=8                   # WICHTIG: Mindestens so viele wie --num_workers im Python-Skript!
#SBATCH --time=08:00:00                     # Laufzeit: 8 Stunden

# --- 1. Umgebung vorbereiten ---
unset SLURM_EXPORT_ENV                  # Saubere Umgebung
module purge                            # Alte Module raus
module load python/pytorch-1.10py3.9    # Basis-Modul

# --- 2. Deine installierten Pakete finden (WICHTIG!) ---
# Da wir tqdm/matplotlib per --user installiert haben, müssen wir Python sagen, wo sie sind:
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

# --- 3. Konfiguration ---
PROJECT_DIR=$(pwd)
DATA_PATH="$WORK/cityscapes_hpc"
PRETRAIN_DATA_PATH="$WORK/cityscapes_hpc"

# Gemeinsamer Run-Name (SLURM Jobname + ID), wird auch ans Python-Skript übergeben
RUN_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
RUN_ROOT="$PROJECT_DIR/trainingoutput/logs/$RUN_NAME"
RUN_LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$RUN_LOG_DIR"

# Stdout/Stderr zusätzlich in den Run-Ordner schreiben
exec > >(tee "$RUN_LOG_DIR/output_${SLURM_JOB_ID}.txt") 2> >(tee "$RUN_LOG_DIR/error_${SLURM_JOB_ID}.txt" >&2)

# --- 4. Debug Infos ---
echo "Starte Training auf Node: $(hostname)"
echo "Nutze GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Datenpfad: $DATA_PATH"

# --- 5. Training Starten ---
# Wir übergeben hier die Argumente, die das Python-Skript erwartet (argparse)
python -u train_script.py \
    --data_dir "$DATA_PATH" \
    --pre_data_dir "$PRETRAIN_DATA_PATH" \
    --output_root "$PROJECT_DIR/trainingoutput" \
    --run_name "$RUN_NAME" \
    --epochs 150 \
    --warmup_epochs 10 \
    --pre_epochs 15 \
    --pre_warmup_epochs 5 \
    --batch_size 32 \
    --lr 0.001 \
    --min_lr 0.00001 \
    --pre_lr 0.001 \
    --pre_min_lr 0.00001 \
    --weight_decay 0.0001 \
    --num_workers 8 \
    --save_every 150 \
    --model "unet" \
    --backbone "b1"

#  -u bei python sorgt dafür, dass Print-Ausgaben sofort im Log landen.
