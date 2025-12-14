#!/bin/bash -l
#SBATCH --job-name=unet_cityscapes
#SBATCH --output=logs/output_%j.txt     # Log-Dateien (Ordner 'logs' muss existieren!)
#SBATCH --error=logs/error_%j.txt

# --- Hardware Ressourcen (Alex Cluster) ---
#SBATCH --partition=a40                 # 'a40' oder 'a100' (je nach Verfügbarkeit)
#SBATCH --gres=gpu:1                    # 1 GPU
#SBATCH --nodes=1                       # 1 Knoten
#SBATCH --ntasks=1                      # 1 Prozess
#SBATCH --cpus-per-task=8               # WICHTIG: Mindestens so viele wie --num_workers im Python-Skript!
#SBATCH --time=08:00:00                 # Laufzeit: 8 Stunden

# --- 1. Umgebung vorbereiten ---
unset SLURM_EXPORT_ENV                  # Saubere Umgebung
module purge                            # Alte Module raus
module load python/pytorch-1.10py3.9    # Dein Basis-Modul

# --- 2. Deine installierten Pakete finden (WICHTIG!) ---
# Da wir tqdm/matplotlib per --user installiert haben, müssen wir Python sagen, wo sie sind:
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH

# --- 3. Konfiguration ---
# Wo liegt dein "Cityscapes" Ordner wirklich?
# Falls er im selben Projektordner unter "01_data" liegt, nutze dies:
PROJECT_DIR=$(pwd)
DATA_PATH="$WORK/cityscapes_intersections_heading/fine_intersections_4"        

# --- 4. Debug Infos ---
echo "Starte Training auf Node: $(hostname)"
echo "Nutze GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Datenpfad: $DATA_PATH"

# --- 5. Training Starten ---
# Wir übergeben hier die Argumente, die dein Python-Skript erwartet (argparse)
python -u train_script.py \
    --data_dir "$DATA_PATH" \
    --epochs 10 \
    --batch_size 8 \
    --lr 0.0001 \
    --num_workers 8 \
    --checkpoint_dir "$PROJECT_DIR/checkpoints" \
    --log_dir "$PROJECT_DIR/logs"

# Das -u bei python sorgt dafür, dass Print-Ausgaben sofort im Log landen.