import torchvision.transforms.functional as F
import torchvision.transforms as T
import random
import numpy as np
from PIL import Image, ImageFilter

"""
Custom Augmentation (Native PyTorch/PIL Implementation)

This implementation was implemented from scratch to avoid external 
dependencies. The structure – and especially the selection and logic 
of the more complex augmentations added later – follows the 
'albumentations' library.
https://albumentations.ai/docs/
"""


class OneOf:
    """Randomly selects one augmentation from a list."""

    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            t = random.choice(self.transforms)
            image, label = t(image, label)
        return image, label

class JointRandomScale:
    """Randomly scale image and label by the same factor."""

    def __init__(self, scale_range=(0.5, 1.0)):
        self.scale_range = scale_range

    def __call__(self, image, label):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        w, h = image.size
        new_w = int(w * scale)
        new_h = int(h * scale)

        image = F.resize(image, (new_h, new_w), interpolation=F.InterpolationMode.BILINEAR)
        label = F.resize(label, (new_h, new_w), interpolation=F.InterpolationMode.NEAREST)
        return image, label


class JointRandomCrop:
    """Crop image and label consistently to target size."""

    def __init__(self, size):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)

    def __call__(self, image, label):
        h, w = self.size
        img_h, img_w = image.size[1], image.size[0]
        label_h, label_w = label.size[1], label.size[0]

        # Ensure both have same size
        assert img_h == label_h and img_w == label_w, "Image and label must have same dimensions"

        # If smaller than target, no crop needed
        if img_h < h or img_w < w:
            return image, label

        top = random.randint(0, img_h - h)
        left = random.randint(0, img_w - w)

        image = F.crop(image, top, left, h, w)
        label = F.crop(label, top, left, h, w)
        return image, label


class JointRandomAffine:
    """Corresponds to A.ShiftScaleRotate: Rotate, Translate, Scale."""

    def __init__(self, degrees=0, translate=None, scale=None, p=0.5, ignore_index=255):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.p = p
        self.ignore_index = ignore_index

    def __call__(self, image, label):
        if random.random() < self.p:
            # Generate parameters
            # Note: degrees must be a range, e.g. (-30, 30)
            angle, translations, scale, shear = T.RandomAffine.get_params(
                degrees=self.degrees if isinstance(self.degrees, (tuple, list)) else (-self.degrees, self.degrees),
                translate=self.translate,
                scale_ranges=self.scale,
                shears=None,
                img_size=image.size
            )

            image = F.affine(image, angle, translations, scale, shear, F.InterpolationMode.BILINEAR)
            # IMPORTANT: Fill mask with ignore_index during rotation
            label = F.affine(label, angle, translations, scale, shear, F.InterpolationMode.NEAREST,
                             fill=self.ignore_index)

        return image, label


class JointRandomHorizontalFlip:
    """Flip image and label horizontally with same probability."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = F.hflip(image)
            label = F.hflip(label)
        return image, label


class RandomBlur:
    """Simulates MotionBlur / MedianBlur / GaussianBlur."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            # Randomly choose a blur type
            mode = random.choice(['gaussian', 'box'])
            if mode == 'gaussian':
                radius = random.uniform(0.5, 2.0)
                image = image.filter(ImageFilter.GaussianBlur(radius))
            else:
                image = image.filter(ImageFilter.BLUR)
        return image, label


class RandomNoise:
    """Adds Gaussian noise."""

    def __init__(self, p=0.5, intensity=20):
        self.p = p
        self.intensity = intensity

    def __call__(self, image, label):
        if random.random() < self.p:
            arr = np.array(image).astype(np.int16)
            noise = np.random.randint(-self.intensity, self.intensity, arr.shape)
            arr = arr + noise
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)
        return image, label


class JointColorJitter:
    """Apply color jitter only to image (not label)."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, label):
        image = F.adjust_brightness(image, brightness_factor=1.0 + random.uniform(-self.brightness, self.brightness))
        image = F.adjust_contrast(image, contrast_factor=1.0 + random.uniform(-self.contrast, self.contrast))
        image = F.adjust_saturation(image, saturation_factor=1.0 + random.uniform(-self.saturation, self.saturation))
        image = F.adjust_hue(image, hue_factor=random.uniform(-self.hue, self.hue))
        return image, label


# https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/169721
class RandomCoarseDropout:
    """
    Cuts out rectangles from the image (Cutout) AND the label.
    The label is filled with the ignore_index.
    """

    def __init__(self, max_holes=8, max_height=32, max_width=32, p=0.5, fill_label=255):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.p = p
        self.fill_label = fill_label  # The value set in the label (e.g. 255 for ignore)

    def __call__(self, image, label):
        if random.random() < self.p:
            w, h = image.size

            # Convert image and label to Numpy arrays
            img_arr = np.array(image)
            lbl_arr = np.array(label)

            num_holes = random.randint(1, self.max_holes)
            for _ in range(num_holes):
                y = random.randint(0, h)
                x = random.randint(0, w)
                h_hole = random.randint(4, self.max_height)
                w_hole = random.randint(4, self.max_width)

                y2 = min(h, y + h_hole)
                x2 = min(w, x + w_hole)

                # Image: Set pixels to black (0)
                # Check if image is 2D (grayscale) or 3D (RGB)
                if len(img_arr.shape) == 3:
                    img_arr[y:y2, x:x2, :] = 0
                else:
                    img_arr[y:y2, x:x2] = 0

                # Label: Set pixels to ignore_index (255)
                # Labels are usually 2D (H, W)
                lbl_arr[y:y2, x:x2] = self.fill_label

            # Back to PIL Images
            image = Image.fromarray(img_arr)
            label = Image.fromarray(lbl_arr)

        return image, label


class JointRandomScale:
    """Rescales image and label by a random factor within scale_range."""

    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, image, label):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        w, h = image.size
        new_size = (int(h * scale), int(w * scale))

        image = F.resize(image, new_size, interpolation=F.InterpolationMode.BILINEAR)
        label = F.resize(label, new_size, interpolation=F.InterpolationMode.NEAREST)
        return image, label


class JointRandomCrop:
    """Crops image and label at a random location to given size."""

    def __init__(self, size):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)

    def __call__(self, image, label):
        i, j, h, w = T.RandomCrop.get_params(image, output_size=self.size)

        image = F.crop(image, i, j, h, w)
        label = F.crop(label, i, j, h, w)
        return image, label


class JointResize:
    """Resize image and label to given size."""

    def __init__(self, size):
        self.size = size if isinstance(size, (tuple, list)) else (size, size)

    def __call__(self, image, label):
        image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
        label = F.resize(label, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, label


class JointToTensor:
    """Convert image and label to tensors."""

    def __call__(self, image, label):
        image = F.pil_to_tensor(image).float() / 255.0
        label = F.pil_to_tensor(label).long().squeeze()
        return image, label


class JointCompose:
    """Apply a sequence of transforms to both image and label."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label