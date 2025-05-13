import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps
import requests
from io import BytesIO

# Load pre-trained model (ResNet-18)
model = models.resnet18(pretrained=True)
model.eval()

# Image transformation
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature extraction function
def get_latent(image):
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.relu(model.bn1(model.conv1(x))))))))
    return features.view(-1)

# Cosine similarity calculation
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# Load a sample image from the web
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Example.jpg/640px-Example.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")
original_latent = get_latent(img)

# Apply image modifications
def apply_modifications(img):
    mods = {}
    mods['original'] = img

    # Random masking
    masked = img.copy()
    mask = Image.new("RGB", (50, 50), (0, 0, 0))
    masked.paste(mask, (60, 60))
    mods['masked'] = masked

    # Gaussian blur
    mods['blur'] = img.filter(ImageFilter.GaussianBlur(4))

    # Gaussian noise
    np_img = np.array(img) / 255.0
    noise = np.random.normal(0, 0.1, np_img.shape)
    noisy = np.clip(np_img + noise, 0, 1)
    mods['noisy'] = Image.fromarray((noisy * 255).astype(np.uint8))

    # Cropping
    mods['cropped'] = img.crop((30, 30, 200, 200)).resize(img.size)

    # Rotation
    mods['rotated'] = img.rotate(20)

    return mods

# Run the experiment
mod_images = apply_modifications(img)
results = {}

for name, mod_img in mod_images.items():
    latent = get_latent(mod_img)
    sim = cosine_similarity(original_latent, latent)
    results[name] = sim

# Plot results
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values())
plt.title("Cosine Similarity with Original Latent Vector")
plt.ylabel("Cosine Similarity")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
