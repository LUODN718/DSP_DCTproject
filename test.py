import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

CLASS_NAMES_ZH = {
    "n01440764": "丁鲷鱼",
    "n02102040": "英国史宾格犬",
    "n02979186": "卡式录音机",
    "n03000684": "电锯",
    "n03028079": "教堂",
    "n03394916": "法国号",
    "n03417042": "垃圾车",
    "n03425413": "加油泵",
    "n03445777": "高尔夫球",
    "n03888257": "降落伞",
}

# 和 train_classifier.py 里保持一致
class TinyCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# 1) 加载 checkpoint
ckpt = torch.load("tiny_imagenette.pt", map_location="cpu")
idx_to_folder = ckpt["idx_to_folder"]

# 2) 构建模型并加载参数
model = TinyCNN(num_classes=len(idx_to_folder))
model.load_state_dict(ckpt["model_state"])
model.eval()

# 3) 预处理（要和训练时一致）
tfm = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

img = Image.open("/Users/lljk/Desktop/image.JPEG").convert("RGB")
x = tfm(img).unsqueeze(0)  # [1, 3, 160, 160]

# 4) 推理
with torch.no_grad():
    logits = model(x)
    pred_idx = logits.argmax(dim=1).item()

pred_folder = idx_to_folder[pred_idx]
pred_name_zh = CLASS_NAMES_ZH.get(pred_folder, pred_folder)
print(f"预测类别: {pred_name_zh} ({pred_folder})")