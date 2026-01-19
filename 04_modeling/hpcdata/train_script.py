import sys
import argparse
from datetime import datetime
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

# Import custom modules
repo_root = Path(__file__).resolve().parent
for p in (repo_root / "04_modeling", repo_root / "03_datapreparation"):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.append(p_str)

from models.deeplab import DeepLabV3Plus
from models.segformer import SegFormer
from models.unet import UNet
from utils.focal_loss import FocalLoss
from cityscapes_dataset import CityscapesDataset
from augmentations import (
    JointCompose, JointToTensor, JointResize, JointRandomHorizontalFlip,
    JointRandomAffine, JointColorJitter, RandomBlur, RandomNoise,
    RandomCoarseDropout, OneOf, JointRandomScale, JointRandomCrop, JointNormalize
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model on Cityscapes")
    parser.add_argument("--data_dir", type=str, required=True, help="Pfad zum Datensatz-Stamm (mit leftImg8bit/gtFine)")
    parser.add_argument("--pre_data_dir", type=str, default=None, help="Pfad zum zusätzlichen Datensatz für Pretraining (mit leftImg8bit_trainextra/gtCoarse)")
    parser.add_argument(
        "--output_root",
        type=str,
        default="./trainingoutput",
        help="Basispfad für Trainingsergebnisse (Run-Ordner wird automatisch erzeugt)",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None, help=argparse.SUPPRESS)  # deprecated
    parser.add_argument("--log_dir", type=str, default=None, help=argparse.SUPPRESS)  # deprecated
    parser.add_argument("--run_name", type=str, default=None, help="Name für Logs/Checkpoints (leer lassen für automatischen Namen)")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--pre_epochs", type=int, default=0, help="Anzahl der Pretraining-Epochen auf dem zusätzlichen Datensatz (0 = kein Pretraining)")
    parser.add_argument("--pre_warmup_epochs", type=int, default=0, help="Anzahl der Warmup-Epochen während des Pretrainings")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--pre_lr", type=float, default=1e-3)
    parser.add_argument("--pre_min_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=50, help="Alle N Epochen Checkpoint schreiben")
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "deeplab", "segformer"],
        help='Welches Modell trainiert werden soll: "unet", "deeplab" oder "segformer"',
    )
    parser.add_argument("--backbone", type=str, default="b0", choices=["b0", "b1", "b2"],
                        help="SegFormer Variante (b0, b1, b2)")
    return parser.parse_args()

### Notebook functions
def compute_miou(pred, truth, num_classes, ignore_index=255):
    ious = []
    pred = pred.flatten()
    truth = truth.flatten()

    mask = truth != ignore_index
    pred = pred[mask]
    truth = truth[mask]

    for clas in range(num_classes):
        pred_clas = pred == clas
        truth_clas = truth == clas
        intersection = (pred_clas & truth_clas).sum().item()
        union = (pred_clas | truth_clas).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious), ious


def compute_accuracy(pred, truth, ignore_index=255):
    mask = truth != ignore_index
    pred = pred[mask]
    truth = truth[mask]
    return (pred == truth).float().mean()


def train(dataloader, model, loss_function, optimizer, device, num_classes, scheduler):
    model.train()
    epoch_loss, epoch_acc, epoch_miou = 0, 0, 0

    pbar = tqdm(dataloader, desc="Training loop", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(images)
        loss = loss_function(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        prediction = output.argmax(dim=1)
        acc = compute_accuracy(prediction, labels)
        miou, _ = compute_miou(prediction, labels, num_classes)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_miou += miou

    num_batches = len(dataloader)
    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_miou / num_batches


def valid(dataloader, model, loss_function, device, num_classes):
    model.eval()
    epoch_loss, epoch_acc, epoch_miou = 0, 0, 0

    pbar = tqdm(dataloader, desc="Validation loop", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            output = model(images)
            loss = loss_function(output, labels)

            prediction = output.argmax(dim=1)
            acc = compute_accuracy(prediction, labels)
            miou, _ = compute_miou(prediction, labels, num_classes)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_miou += miou

    num_batches = len(dataloader)
    return epoch_loss / num_batches, epoch_acc / num_batches, epoch_miou / num_batches


def prepare_output_dirs(model_name, output_root, run_name=None):
    """
    Create sibling run folders inside trainingoutput/checkpoints and trainingoutput/logs.
    Both use the same run name <model>_<YYYYMMDD_HHMMSS>.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"{model_name}_{timestamp}"

    checkpoints_dir = output_root / "checkpoints" / run_name
    logs_root = output_root / "logs" / run_name
    tensorboard_dir = logs_root / "tensorboard"
    logs_errors_dir = logs_root / "logs_errors"

    for path in (checkpoints_dir, tensorboard_dir, logs_errors_dir):
        path.mkdir(parents=True, exist_ok=True)

    return run_name, checkpoints_dir, tensorboard_dir, logs_errors_dir, logs_root


def build_model(model_name, num_classes, backbone="b0"):
    name = model_name.lower()
    if name == "unet":
        return UNet(num_classes)
    if name == "deeplab":
        return DeepLabV3Plus(num_classes)
    if name == "segformer":
        return SegFormer(num_classes, phi=backbone)
    raise ValueError(f"Unsupported model: {model_name}")

def build_warmup_cosine_scheduler(optimizer, base_lr, min_lr, total_iters, warmup_iters):
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,      # starts at 1% of base lr
        total_iters=warmup_iters
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_iters - warmup_iters,
        eta_min=min_lr
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters]
    )
    return scheduler


if __name__ == "__main__":
    args = parse_args()

    data_dir = Path(args.data_dir)
    image_dir = data_dir / "leftImg8bit_trainvaltest/leftImg8bit"
    label_dir = data_dir / "gtFine_trainvaltest/gtFine"

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    (
        run_name_generated,
        checkpoint_dir,
        tensorboard_dir,
        logs_errors_dir,
        logs_root,
    ) = prepare_output_dirs(args.model, output_root, run_name=args.run_name)
    
    # Logging all arguments
    print("-" * 30)
    print("CONFIGURATION / ARGUMENTS:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-" * 30)

    print(f"Run name: {run_name_generated}")
    print(f"Checkpoints dir: {checkpoint_dir}")
    print(f"TensorBoard dir: {tensorboard_dir}")
    print(f"Logs/errors dir: {logs_errors_dir}")

    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # Save arguments to TensorBoard
    arg_dict = vars(args)
    arg_str = "\n".join([f"{arg}: {value}" for arg, value in arg_dict.items()])
    writer.add_text("Configuration", arg_str)

    # Save arguments to txt file
    with open(logs_root / "args.txt", "w") as f:
        f.write(arg_str)

    ### Transforms (like in notebook)
    # Crop-size for training
    crop_height, crop_width = 512, 1024
    # Resize-size for validation
    height, width = 512, 1024

    # Joint transforms for training with augmentations
    joint_train_transforms = JointCompose([
        JointRandomScale(scale_range=(0.5, 1.0)),
        JointRandomCrop(size=(crop_height, crop_width)),
        JointRandomHorizontalFlip(p=0.5),
        JointColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        JointToTensor(),
    ])

    # Joint transforms for validation (only resize)
    joint_valid_transforms = JointCompose([
        JointResize(size=(height, width)),
        JointToTensor(),
    ])

    def joint_train_transform(image, label):
        image, label = Image.fromarray(image), Image.fromarray(label)
        image, label = joint_train_transforms(image, label)
        return image, label

    def joint_valid_transform(image, label):
        image, label = Image.fromarray(image), Image.fromarray(label)
        image, label = joint_valid_transforms(image, label)
        return image, label

    # Datasets + Loader
    train_dataset = CityscapesDataset(
        image_dir, label_dir, split="train", joint_transform=joint_train_transform
    )
    valid_dataset = CityscapesDataset(
        image_dir, label_dir, split="val", joint_transform=joint_valid_transform
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Setup the model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    num_classes = 20
    model = build_model(args.model, num_classes, backbone=args.backbone).to(device)
    model_name = getattr(model, "name", args.model)
    checkpoint_prefix = run_name_generated

    ### Pretraining
    if args.pre_data_dir is not None and args.pre_epochs > 0:
        pre_data_dir = Path(args.pre_data_dir)
        pre_image_dir = pre_data_dir / "leftImg8bit_trainextra/leftImg8bit"
        pre_label_dir = pre_data_dir / "gtCoarse/gtCoarse"

        # Datasets + Loader
        pre_dataset = CityscapesDataset(
            pre_image_dir, pre_label_dir, split="train_extra", joint_transform=joint_train_transform
        )

        pre_dataloader = DataLoader(
            pre_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )

        # Setup loss_function, optimizer, scheduler for pretraining
        pre_loss_function = FocalLoss(alpha=1, gamma=2)
        pre_optimizer = optim.AdamW(params=model.parameters(), lr=args.pre_lr, weight_decay=args.weight_decay)

        pre_iters_per_epoch = len(pre_dataloader)
        pre_warmup_iters = args.pre_warmup_epochs * pre_iters_per_epoch
        pre_total_iters = args.pre_epochs * pre_iters_per_epoch
        pre_scheduler = build_warmup_cosine_scheduler(optimizer=pre_optimizer, base_lr=args.pre_lr, min_lr=args.pre_min_lr, total_iters=pre_total_iters, warmup_iters=pre_warmup_iters)

        pbar = tqdm(range(args.pre_epochs))
        for epoch in pbar:
            pbar.set_description(f"PRETRAINING: Epoch {epoch+1}/{args.pre_epochs}")

            train_loss, train_acc, train_miou = train(
                pre_dataloader, model, pre_loss_function, pre_optimizer, device, num_classes, pre_scheduler
            )

            pbar.set_postfix({
                "Train Loss": f"{train_loss:.3f}", 
                "Train Acc": f"{train_acc:.3f}",   
                "Train MIoU": f"{train_miou:.3f}"
            })

            # Save final pretraining checkpoint
            if (epoch + 1) % args.pre_epochs == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": pre_optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_miou": train_miou
                }

                torch.save(checkpoint, checkpoint_dir / f"{checkpoint_prefix}_pretrain_final_epoch_{epoch}.pth")

            current_lr = pre_optimizer.param_groups[0]["lr"]
            writer.add_scalar("PRETRAINING/lr", current_lr, global_step=epoch)
            writer.add_scalar("PRETRAINING/loss", train_loss, global_step=epoch)
            writer.add_scalar("PRETRAINING/acc", train_acc, global_step=epoch)
            writer.add_scalar("PRETRAINING/miou", train_miou, global_step=epoch)
            writer.flush()

    ### Training    
    loss_function = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    iters_per_epoch = len(train_dataloader)
    warmup_iters = args.warmup_epochs * iters_per_epoch
    total_iters = args.epochs * iters_per_epoch
    scheduler = build_warmup_cosine_scheduler(optimizer=optimizer, base_lr=args.lr, min_lr=args.min_lr, total_iters=total_iters, warmup_iters=warmup_iters)

    best_val_miou = 0.0

    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        pbar.set_description(f"TRAINING: Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, train_miou = train(
            train_dataloader, model, loss_function, optimizer, device, num_classes, scheduler
        )
        val_loss, val_acc, val_miou = valid(
            valid_dataloader, model, loss_function, device, num_classes
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_miou": train_miou,
            "val_miou": val_miou,
        }

        if (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, checkpoint_dir / f"{checkpoint_prefix}_epoch_{epoch}.pth")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(checkpoint, checkpoint_dir / f"best_{checkpoint_prefix}.pth")

        pbar.set_postfix({
                "Train Loss": f"{train_loss:.3f}",
                "Val Loss": f"{val_loss:.3f}",
                "Train Acc": f"{train_acc:.3f}",
                "Val Acc": f"{val_acc:.3f}",
                "Train MIoU": f"{train_miou:.3f}",
                "Val MIoU": f"{val_miou:.3f}",
            })
        
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("TRAINING/lr", current_lr, global_step=epoch)
        writer.add_scalars("TRAINING/loss", {"train loss": train_loss, "val loss": val_loss}, global_step=epoch)
        writer.add_scalars("TRAINING/acc", {"train acc": train_acc, "val acc": val_acc}, global_step=epoch)
        writer.add_scalars("TRAINING/MIoU", {"train MIoU": train_miou, "val MIoU": val_miou}, global_step=epoch)
        writer.flush()

    writer.close()
