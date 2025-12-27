import random
import io
from PIL import Image
from torchvision import transforms

# --- JPEG compression simulation ---
class RandomJPEGCompression:
    def __init__(self, quality_range=(30, 95), p=0.4):  # ↓ reduced p
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buffer = io.BytesIO()
        quality = random.randint(*self.quality_range)
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


# --- Training transforms ---
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),

    # Simulate camera resolution & resampling pipeline
    transforms.RandomApply([
        transforms.Resize(200),
        transforms.Resize(224),
    ], p=0.3),

    # Handle JPG (train) ↔ PNG (test) domain gap
    RandomJPEGCompression(quality_range=(30, 95), p=0.4),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=7),

    # Reduce color dependency (spoof shortcut)
    transforms.RandomGrayscale(p=0.1),

    # PNG images are sharper → simulate sharpness variance
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),

    # Camera blur / focus variation
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))
    ], p=0.2),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --- Validation transforms (clean, no augmentation) ---
val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
