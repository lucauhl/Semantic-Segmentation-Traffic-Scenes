from PIL import Image
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
# Add project directories so local modules import correctly
import sys
from pathlib import Path

cwd = Path.cwd().resolve()
repo_root = None
for cand in [cwd, *cwd.parents]:
    if (cand / "04_modeling").exists() and (cand / "03_datapreparation").exists():
        repo_root = cand
        break

if repo_root is None:
    repo_root = cwd

for p in (repo_root / "04_modeling", repo_root / "03_datapreparation"):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.append(p_str)
# Models
from models.unet import UNet
from models.deeplab import DeepLabV3Plus
from models.segformer import SegFormer


### Helpers
# Crop size (Original 1024x2048)
HEIGHT, WIDTH = 512, 1024
# Cityscapes 20 class colors (R,G,B) based on current trainIds (added vegetation and parking classes)
CITYSCAPES_COLORS_20 = np.array([
    [128, 64,128],  # 0: road
    [244, 35,232],  # 1: sidewalk
    [250,170,160],  # 2: parking
    [230,150,140],  # 3: rail track
    [ 70, 70, 70],  # 4: building
    [180,165,180],  # 5: guard rail
    [153,153,153],  # 6: pole
    [250,170, 30],  # 7: traffic light
    [220,220,  0],  # 8: traffic sign
    [107,142, 35],  # 9: vegetation
    [152,251,152],  # 10: terrain
    [ 70,130,180],  # 11: sky
    [220, 20, 60],  # 12: person
    [255,  0,  0],  # 13: rider
    [  0,  0,142],  # 14: car
    [  0,  0, 70],  # 15: truck
    [  0, 60,100],  # 16: bus
    [  0, 80,100],  # 17: train
    [  0,  0,230],  # 18: motorcycle
    [119, 11, 32],  # 19: bicycle
], dtype=np.uint8)

def label_to_color(label, color_array):
    h, w = label.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(color_array):
        mask = label == class_idx
        color_image[mask] = color

    # Set ignore_index to black
    mask = label == 255
    color_image[mask] = [0, 0, 0]
    return color_image

def color_to_label(color_image, color_array):
    h, w = color_image.shape[:2]
    label = np.full((h, w), 255, dtype=np.uint8)
    # Check for black pixels first (ignore_index)
    black_mask = np.all(color_image == [0, 0, 0], axis=-1)
    # Match each color to its class index
    for class_idx, color in enumerate(color_array):
        mask = np.all(color_image == color, axis=-1)
        label[mask] = class_idx
    # Ensure black pixels stay as 255
    label[black_mask] = 255
    return label

### Evaluation Criteria
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

### Model
def build_model(model_name, num_classes, backbone="b1"):
    name = model_name.lower()
    if name == "unet":
        return UNet(num_classes)
    if name == "deeplab":
        return DeepLabV3Plus(num_classes)
    if name == "segformer":
        return SegFormer(num_classes, phi=backbone)
    raise ValueError(f"Unsupported model: {model_name}")


def load_model(name, checkpoint, num_classes=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint_path = f"../checkpoints/{checkpoint}/best_{checkpoint}.pth"
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build model and move to device
    model = build_model(name, num_classes)
    model_name = getattr(model, "name")
    model = model.to(device)

    # Load weights
    model.load_state_dict(checkpoint_data["model_state_dict"])
    # Switch model to eval mode
    model.eval()
    
    print(f"Loaded model: {model_name}")
    print(f"Device: {device}")
    print(f"Best checkpoint:")
    print(f" - Epoch: {checkpoint_data['epoch']:.3f}")
    print(f" - Train Loss: {checkpoint_data['train_loss']:.3f}")
    print(f" - Train Accuracy: {checkpoint_data['train_acc']:.3f}")
    print(f" - Train MIoU: {checkpoint_data['train_miou']:.3f}")
    print(f" - Val Loss: {checkpoint_data['val_loss']:.3f}")
    print(f" - Val Accuracy: {checkpoint_data['val_acc']:.3f}")
    print(f" - Val MIoU: {checkpoint_data['val_miou']:.3f}")

    # Extract training stats
    train_stats = {
        "epoch": checkpoint_data.get('epoch', 0),
        "train_miou": checkpoint_data.get('train_miou', 0.0),
        "train_acc": checkpoint_data.get('train_acc', 0.0),
        "val_miou": checkpoint_data.get('val_miou', 0.0),
        "val_acc": checkpoint_data.get('val_acc', 0.0),
    }

    return model, device, train_stats

### Inference, Evaluation and Visualization
def infer(model, device, image):
    # Transforms 
    def image_transform(image):
        image = Image.fromarray(image)  # numpy -> PIL
        image = F.resize(image, (HEIGHT, WIDTH), interpolation=F.InterpolationMode.BILINEAR) # Bilinear for RGB-Images
        image = F.pil_to_tensor(image).float() / 255.0  # Scaling
        return image
    
    with torch.no_grad():
        image = image_transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)

        prediction = model(image)
        prediction = prediction.squeeze(0)
        predicted = torch.argmax(prediction, dim=0).cpu()

    return predicted

def eval(pred, truth, num_classes):
    # Transforms 
    def label_transform(label):
        label = Image.fromarray(label)  # numpy -> PIL
        label = F.resize(label, (HEIGHT, WIDTH), interpolation=F.InterpolationMode.NEAREST) # Nearest Neighbor for Labels
        label = F.pil_to_tensor(label).long().squeeze()  # PIL -> LongTensor no Scaling, squeeze() to remove channel dimension
        return label
    
    train_label = color_to_label(truth, CITYSCAPES_COLORS_20)
    train_label = label_transform(train_label)
    miou, ious = compute_miou(pred, train_label, num_classes)
    accuracy = compute_accuracy(pred, train_label)

    return miou, ious, accuracy

def print_results(miou, acc):
    print("Results:")
    print(f" - MIoU: {miou:.3f}")
    print(f" - Pixel Accuracy: {acc:.3f}")

def plot_images(image, image_title, label, label_title, figsize=(12, 6)):
    _, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].imshow(image)
    ax[0].set_title(image_title)
    ax[0].axis("off")

    ax[1].imshow(label)
    ax[1].set_title(label_title)
    ax[1].axis("off")

    plt.show()

# Visualization (3 Plots)
def plot_three_images(img1, title1, img2, title2, img3, title3, figsize=(18, 6)):
    _, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(img1); ax[0].set_title(title1); ax[0].axis("off")
    ax[1].imshow(img2); ax[1].set_title(title2); ax[1].axis("off")
    ax[2].imshow(img3); ax[2].set_title(title3); ax[2].axis("off")
    plt.show()


def run_evaluation_pipeline(model_type, checkpoint_name, base_path="./CVATAnnotations", show_plots=True):
    cvat_path = Path(base_path)
    if not cvat_path.exists():
        print(f"Path {base_path} does not exist.")
        return None

    # Load model and saved stats
    model, device, train_stats = load_model(model_type, checkpoint_name)

    image_files = sorted(list(cvat_path.glob("*r.jpg")))

    # Lists for eval results
    eval_mious = []
    eval_accs = []

    for image_path in image_files:
        label_name = image_path.name.replace("r.jpg", ".png")
        label_path = cvat_path / label_name

        if not label_path.exists(): continue

        image = cv2.imread(str(image_path))
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        rgb_label = cv2.imread(str(label_path))
        if rgb_label is None: continue
        rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_BGR2RGB)

        prediction = infer(model, device, image)
        miou, _, acc = eval(prediction, rgb_label, num_classes=20)

        eval_mious.append(miou)
        eval_accs.append(acc)

        if show_plots:
            rgb_pred = label_to_color(prediction.numpy(), CITYSCAPES_COLORS_20)
            plot_three_images(
                image, f"Original: {image_path.name}",
                rgb_label, "Ground Truth",
                rgb_pred, f"Pred ({model_type})"
            )

    # Compute averages (eval) for Regensburg
    avg_eval_miou = np.mean(eval_mious) if eval_mious else 0.0
    avg_eval_acc = np.mean(eval_accs) if eval_accs else 0.0

    # Create summary dictionary
    summary = {
        "Model": model_type,
        "Checkpoint": checkpoint_name,
        "Train MIoU": train_stats['train_miou'],
        "Train Acc": train_stats['train_acc'],
        "Val (Cityscapes) MIoU": train_stats['val_miou'],
        "Val (Cityscapes) Acc": train_stats['val_acc'],
        "Eval (Regensburg) MIoU": avg_eval_miou,
        "Eval (Regensburg) Acc": avg_eval_acc
    }

    return summary
