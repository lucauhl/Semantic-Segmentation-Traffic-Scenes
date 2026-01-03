import torchvision.transforms.functional as F
import random


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


class JointRandomHorizontalFlip:
    """Flip image and label horizontally with same probability."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = F.hflip(image)
            label = F.hflip(label)
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