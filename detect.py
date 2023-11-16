import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from tqdm import tqdm
from PIL import Image

# 하이퍼파라미터
batch_size = 64
epochs = 30
learning_rate = 0.001
img_size = 256

# GPU 사용 여부 확인
device = torch.device("mps")


# 이미지 데이터셋 클래스 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(img_size // 2 * img_size // 2 * 16, 101)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 모델 불러오기
model = SimpleCNN()
model.load_state_dict(torch.load("trained_model.pth"))
model.to(device)
model.eval()

# 이미지 전처리
transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
)

# 이미지 불러오기
image = Image.open("test.jpg").convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# 예측
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)

print(predicted.item())
