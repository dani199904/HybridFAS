# ============================================================
# SKT-Live V2.5 — FULL INFERENCE PIPELINE (TEST SET)
# ============================================================

import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------- 1. Architecture Components --------------------

class PatchEmbedV2_3(nn.Module):
    def __init__(self, in_ch=3, embed_dim=256):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, embed_dim, kernel_size=16, stride=16)
    def forward(self, x): return self.conv(x)

class SpoofEdgeCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.b1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.b2 = nn.BatchNorm2d(dim)
    def forward(self, x): return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))

class SpoofReflectCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 5, padding=2)
        self.b1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 5, padding=2)
        self.b2 = nn.BatchNorm2d(dim)
    def forward(self, x): return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))

class SpoofNoiseCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.b1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.b2 = nn.BatchNorm2d(dim)
    def forward(self, x): return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))

class SpoofBranchV2_3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge, self.reflect, self.noise = SpoofEdgeCNN(dim), SpoofReflectCNN(dim), SpoofNoiseCNN(dim)
    def forward(self, x): return (self.edge(x) + self.reflect(x) + self.noise(x)) / 3

class RealSkinCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 5, padding=2)
        self.b1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 5, padding=2)
        self.b2 = nn.BatchNorm2d(dim)
    def forward(self, x): return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))

class RealIllumCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 7, padding=3)
        self.b1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 7, padding=3)
        self.b2 = nn.BatchNorm2d(dim)
    def forward(self, x): return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))

class RealStructureCNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv2d(dim, dim, 5, padding=2)
        self.b1 = nn.BatchNorm2d(dim)
        self.c2 = nn.Conv2d(dim, dim, 5, padding=2)
        self.b2 = nn.BatchNorm2d(dim)
    def forward(self, x): return self.b2(self.c2(F.relu(self.b1(self.c1(x)))))

class RealBranchV2_3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.skin, self.illum, self.struct = RealSkinCNN(dim), RealIllumCNN(dim), RealStructureCNN(dim)
    def forward(self, x): return (self.skin(x) + self.illum(x) + self.struct(x)) / 3

class FusionGateV2_3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c1, self.c2, self.drop = nn.Conv2d(dim * 3, dim, 1), nn.Conv2d(dim, 3, 1), nn.Dropout2d(0.1)
    def forward(self, base, spoof, real):
        x = F.relu(self.c1(torch.cat([base, spoof, real], dim=1)))
        w = F.softmax(self.c2(self.drop(x)), dim=1)
        return w[:, 0:1] * base + w[:, 1:2] * spoof + w[:, 2:3] * real

class TransformerEncoderV2_3(nn.Module):
    def __init__(self, dim=256, layers=4, heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
    def forward(self, x): return self.encoder(x)

class SKTLiveV2_5(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.patch = PatchEmbedV2_3(3, embed_dim)
        self.spoof, self.real, self.fusion = SpoofBranchV2_3(embed_dim), RealBranchV2_3(embed_dim), FusionGateV2_3(embed_dim)
        self.norm, self.cls = nn.LayerNorm(embed_dim), nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.transformer = TransformerEncoderV2_3(embed_dim)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))
    
    def forward(self, x):
        base = self.patch(x)
        fused = F.gelu(self.fusion(base, self.spoof(base), self.real(base)))
        tokens = self.norm(fused.flatten(2).transpose(1, 2))
        tokens = torch.cat([self.cls.expand(tokens.size(0), -1, -1), tokens], dim=1)
        return self.head(self.transformer(tokens)[:, 0])

# -------------------- 2. Setup --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "skt_live_best_model_v2.5.pth"
test_data_root = "/media/danial-ahmed/HDD/Processed_CelebA_Spoof/test"

model = SKTLiveV2_5().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Model Loaded from {model_path}")

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------- 3. Inference Loop --------------------
samples = []
for cls_name, label in [('live', 0), ('spoof', 1)]:
    path = os.path.join(test_data_root, cls_name)
    for img_p in glob.glob(os.path.join(path, "*")):
        samples.append((img_p, label))

print(f"🔍 Testing on {len(samples)} samples...")
results = []
with torch.no_grad():
    for path, label in tqdm(samples):
        img = Image.open(path).convert("RGB")
        img_t = test_tfms(img).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(img_t)).item()
        results.append({"prob": prob, "label": label, "pred": 1 if prob > 0.5 else 0})

# -------------------- 4. Report & Metrics --------------------
df = pd.DataFrame(results)
cm = confusion_matrix(df['label'], df['pred'])
tn, fp, fn, tp = cm.ravel()

# Metrics matching your training log nomenclature
acc = (tp + tn) / len(df)
apcer = fp / (fp + tn + 1e-7)  # Live as Spoof (BPCER in ISO)
bpcer = fn / (fn + tp + 1e-7)  # Spoof as Live (APCER in ISO)

print(f"\n" + "="*30 + "\nTEST RESULTS\n" + "="*30)
print(f"Accuracy: {acc:.4f}\nAPCER (Live Error): {apcer:.4f}\nBPCER (Spoof Error): {bpcer:.4f}")

# Visualization
disp = ConfusionMatrixDisplay(cm, display_labels=['live', 'spoof'])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"V2.5 Test Results\nAcc: {acc:.4f}")
plt.savefig("test_confusion_matrix.png")
df.to_csv("test_predictions_v2.5.csv", index=False)
plt.show()
