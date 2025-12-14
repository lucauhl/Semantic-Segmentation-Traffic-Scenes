import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# WICHTIG: Matplotlib "headless" machen, bevor pyplot importiert wird
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# tqdm für Skripte (nicht notebook)
from tqdm import tqdm

# Eigene Module importieren (angenommen, sie liegen im selben Ordner oder PYTHONPATH ist gesetzt)
# Auf dem HPC setzt man den Pfad oft eher über das Slurm-Skript (export PYTHONPATH=$PYTHONPATH:...)
# aber das hier kann bleiben, wenn die Ordnerstruktur gleich bleibt.
repo_root = Path(__file__).resolve().parent # Besser als cwd()
for p in (repo_root / "04_modeling", repo_root / "03_datapreparation"):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.append(p_str)

from models.unet import UNet
from utils.focal_loss import FocalLoss
from cityscapes_dataset import CityscapesDataset

# --- Argument Parser für flexible Pfade ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet on HPC")
    # Pfade sind jetzt Argumente, keine hardcodierten Strings mehr
    parser.add_argument("--data_dir", type=str, required=True, help="Pfad zum Datensatz")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Wo sollen Modelle gespeichert werden?")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Tensorboard Logs")
    
    # Hyperparameter
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8) # 2 ist oft zu klein für HPC GPUs (A100/V100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4, help="CPU Threads für Dataloader")
    
    return parser.parse_args()

# (Helper Functions compute_miou, compute_accuracy, train, valid bleiben fast gleich, 
# aber tqdm sollte angepasst werden, z.B. mininterval=60 setzen, damit Logs nicht explodieren)

def train(dataloader, model, loss_function, optimizer, device, num_classes):
    model.train()
    epoch_loss, epoch_acc, epoch_miou = 0, 0, 0
    
    # mininterval=60 sorgt dafür, dass nur alle 60sek eine Zeile ins Log geschrieben wird
    pbar = tqdm(dataloader, desc="Training", leave=False, mininterval=30) 
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True) # non_blocking hilft beim Speed
        labels = labels.to(device, non_blocking=True)

        output = model(images)
        loss = loss_function(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = output.argmax(dim=1)
        # Metriken berechnen (kann man auch seltener machen um CPU zu sparen)
        acc = (prediction == labels).float().mean() # Vereinfacht für Speed
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    num_batches = len(dataloader)
    return epoch_loss/num_batches, epoch_acc/num_batches, 0 # miou hier evtl weglassen und nur in valid machen für Speed

def valid(dataloader, model, loss_function, device, num_classes):
    model.eval()
    epoch_loss = 0
    # ... Validierungslogik ...
    # Tipp: Validierung auf dem Cluster dauert oft lange. Evtl. nur alle 5 Epochen validieren?
    return 0, 0, 0 # Platzhalter

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    
    # Pfade aus Argumenten nutzen
    data_dir = Path(args.data_dir)
    image_dir = data_dir / "leftImg8bit_trainvaltest/leftImg8bit"
    label_dir = data_dir / "gtFine_trainvaltest/gtFine"
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Transforms
    height, width = 512, 1024
    def image_transform(image):
        image = Image.fromarray(image)
        image = F.resize(image, (height, width), interpolation=F.InterpolationMode.BILINEAR)
        image = F.pil_to_tensor(image).float() / 255.0
        return image

    def label_transform(label):
        label = Image.fromarray(label)
        label = F.resize(label, (height, width), interpolation=F.InterpolationMode.NEAREST)
        label = F.pil_to_tensor(label).long().squeeze()
        return label

    # Datasets
    train_dataset = CityscapesDataset(image_dir, label_dir, split="train", image_transform=image_transform, label_transform=label_transform)
    # Tipp: Nutze pin_memory=True für schnellere Übertragung zur GPU
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, pin_memory=True)
    
    # Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    model = UNet(19).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_function = FocalLoss(gamma=1, ignore_index=255)
    writer = SummaryWriter(log_dir=args.log_dir)

    # Loop
    for epoch in range(args.epochs):
        print(f"Starting Epoch {epoch+1}")
        train_loss, train_acc, _ = train(train_dataloader, model, loss_function, optimizer, device, 19)
        
        # Logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        print(f"Epoch {epoch+1} finished. Loss: {train_loss:.4f}")
        
        # Checkpointing
        if (epoch + 1) % 50 == 0: # Nicht zu oft speichern, das kostet Zeit/Platz
            torch.save(model.state_dict(), checkpoint_dir / f"model_ep{epoch+1}.pth")

    writer.close()