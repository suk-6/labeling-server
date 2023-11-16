import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms, datasets
from tqdm import tqdm
from PIL import Image

# 하이퍼파라미터
batch_size = 64
epochs = 100
learning_rate = 0.001
img_size = 256

# GPU 사용 여부 확인
device = torch.device("mps")


# 이미지 데이터셋 클래스 정의
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, range=None):
        self.root_dir = root_dir
        self.transform = transform
        self.range = range
        self.images = []
        self.labels = []

        # 이미지 폴더에서 파일 리스트 생성
        for filename in os.listdir(root_dir):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                self.images.append(os.path.join(root_dir, filename))
                # 파일명을 '.'으로 분리하여 label 생성
                label = int(filename.split(".")[1])
                self.labels.append(label)

    def __len__(self):
        if self.range is not None:
            count = 0
            for label in self.labels:
                if len(self.range) == 2:
                    if self.range[0] <= label <= self.range[1]:
                        count += 1
                elif len(self.range) == 4:
                    if (
                        self.range[0] <= label <= self.range[1]
                        or self.range[2] <= label <= self.range[3]
                    ):
                        count += 1
            return count
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        if self.range is not None:
            if len(self.range) == 2:
                if self.range[0] <= label <= self.range[1]:
                    pass
                else:
                    return self.__getitem__(idx + 1)
            elif len(self.range) == 4:
                if (
                    self.range[0] <= label <= self.range[1]
                    or self.range[2] <= label <= self.range[3]
                ):
                    pass
                else:
                    return self.__getitem__(idx + 1)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# 데이터 전처리 및 증강
transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
)

augment_transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ]
)

# 데이터셋 생성 및 분할
dataset = CustomDataset(root_dir="images", transform=transform)
augmented_dataset = CustomDataset(root_dir="images", transform=augment_transform)
biased_dataset = CustomDataset(
    root_dir="images", transform=transform, range=(0, 40, 60, 100)
)
dataset = ConcatDataset([dataset, augmented_dataset, biased_dataset])

# 데이터셋을 train과 validation으로 분할 (80% 학습, 20% 검증)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# 간단한 CNN 모델 정의
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


# 모델 및 손실 함수, 최적화 함수 정의
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    # 평균 손실 계산
    train_loss = train_loss / len(train_loader.dataset)

    # 검증 루프
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in tqdm(
            val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
        ):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    # 평균 검증 손실 및 정확도 계산
    val_loss = val_loss / len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print(
        f"Epoch {epoch + 1}/{epochs} => "
        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy * 100:.2f}%"
    )

# 학습이 완료된 모델 저장
torch.save(model.state_dict(), "trained_model.pth")
