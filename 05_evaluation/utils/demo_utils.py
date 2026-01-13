# Image libraries
import numpy as np
from PIL import Image
import cv2
# PyTorch
import torch
import torchvision.transforms.functional as F
# Add project directories so local modules import correctly
import sys
from pathlib import Path
repo_root = Path.cwd()
for p in (repo_root / "../04_modeling", repo_root / "../03_datapreparation"):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.append(p_str)
# Models
from models.unet import UNet
from models.deeplab import DeepLabV3Plus
from models.segformer import SegFormer
# Visualization
import matplotlib.pyplot as plt

### Helpers
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
    # Wenn CUDA verfügbar ist, wird 'cuda' genutzt, sonst 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. Checkpoint laden
    # WICHTIG: map_location=device sorgt dafür, dass die Gewichte auf das richtige Gerät (z.B. CPU) gemappt werden
    checkpoint_path = f"../checkpoints/{checkpoint}/best_{checkpoint}.pth"
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 3. Modell bauen und auf das Gerät schieben
    model = build_model(name, num_classes)
    model_name = getattr(model, "name")
    model = model.to(device)

    # 4. Gewichte laden
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    
    print(f"Loaded model: {model_name}")
    print(f"Best checkpoint:")
    print(f" - Epoch: {checkpoint_data['epoch']}")
    print(f" - Train Loss: {checkpoint_data['train_loss']}")
    print(f" - Train Accuracy: {checkpoint_data['train_acc']}")
    print(f" - Train MIoU: {checkpoint_data['train_miou']}")
    print(f" - Val Loss: {checkpoint_data['val_loss']}")
    print(f" - Val Accuracy: {checkpoint_data['val_acc']}")
    print(f" - Val MIoU: {checkpoint_data['val_miou']}")

    return model, device

### Inference, Evaluation and Visualization
def infer(model, device, image):
    # Transforms 
    height, width = 768, 1024 # Original 1536, 2048
    def image_transform(image):
        image = Image.fromarray(image)  # numpy -> PIL
        image = F.resize(image, (height, width), interpolation=F.InterpolationMode.BILINEAR) # Bilinear for RGB-Images
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
    height, width = 768, 1024 # Original 1536, 2048
    def label_transform(label):
        label = Image.fromarray(label)  # numpy -> PIL
        label = F.resize(label, (height, width), interpolation=F.InterpolationMode.NEAREST) # Nearest Neighbor for Labels
        label = F.pil_to_tensor(label).long().squeeze()  # PIL -> LongTensor no Scaling, squeeze() to remove channel dimension
        return label
    
    train_label = color_to_label(truth, CITYSCAPES_COLORS_20)
    train_label = label_transform(train_label)
    miou, ious = compute_miou(pred, train_label, num_classes)
    accuracy = compute_accuracy(pred, train_label)

    return miou, ious, accuracy

def plot_images(image, image_title, label, label_title, figsize=(12, 6)):
    _, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].imshow(image)
    ax[0].set_title(image_title)
    ax[0].axis("off")

    ax[1].imshow(label)
    ax[1].set_title(label_title)
    ax[1].axis("off")

    plt.show()

