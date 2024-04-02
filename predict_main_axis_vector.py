import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

np.random.seed(12)

# 가상의 데이터셋 생성
class PolygonDataset(Dataset):
    def __init__(self, num_samples=1024, num_vertices=6, img_size=32):
        self.num_samples = num_samples
        self.num_test_samples = 128
        self.num_vertices = num_vertices
        self.img_size = img_size
        self.datasets = []
        self.vecs = []

        self.test_datasets = []
        self.test_vecs = []

        self.make_datasets(self.num_samples, is_test=False)
        self.make_datasets(self.num_test_samples, is_test=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.datasets[idx]

    def get_vec(self, idx):
        return self.vecs[idx]

    def make_each_dataset(self):
        # 각 꼭짓점의 각도를 무작위로 생성
        angles = np.sort(np.random.rand(self.num_vertices) * 2 * math.pi)
        radius = self.img_size / 2 - 1  # 원의 반지름 설정
        center = self.img_size / 2  # 이미지 중앙을 원의 중심으로 설정

        # 원의 주위에 꼭짓점 배치
        vertices = np.array([
            (math.cos(angle) * radius + center, math.sin(angle) * radius + center)
            for angle in angles
        ])

        # 결과에 해당할 것으로 예상되는 벡터 - label 로 사용
        polygon = Polygon(vertices)
        coords = polygon.minimum_rotated_rectangle.exterior.coords
        vecs = [(coords[i + 1][0] - coord[0], coords[i + 1][1] - coord[1]) for i, coord in enumerate(coords[:-1])]
        vec_raw = [vec for vec in vecs if vec[0] > 0 and vec[1] > 0][0]
        vec_length = np.linalg.norm(vec_raw)
        vec = tuple(float(x / vec_length) for x in vec_raw)

        # 이미지 생성
        img = np.zeros((self.img_size, self.img_size))
        for i in range(self.num_vertices - 1):
            pt1 = tuple((vertices[i]).astype(int))
            pt2 = tuple((vertices[i + 1]).astype(int))
            cv2.line(img, pt1, pt2, color=1, thickness=1)
        # 마지막 점과 첫 번째 점을 연결
        pt1 = tuple((vertices[self.num_vertices - 1]).astype(int))
        pt2 = tuple((vertices[0]).astype(int))
        cv2.line(img, pt1, pt2, color=1, thickness=1)

        # PyTorch Tensor로 변환
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0)  # 채널 차원 추가
        return img_tensor, vec

    def make_datasets(self, num_samples, is_test):
        for _ in range(num_samples):
            img_tensor, vec = self.make_each_dataset()

            if is_test:
                self.test_datasets.append(img_tensor)
                self.test_vecs.append(vec)
            else:
                self.datasets.append(img_tensor)
                self.vecs.append(vec)

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # 2차원으로 출력 (가장 지배적인 축의 방향)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# PolygonDataset 인스턴스 시각화 함수
def visualize_polygon_dataset(img_tensors, vecs, comparison_vecs, num_images=5):

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))

    for i in range(num_images):
        img_tensor = img_tensors[i]  # 데이터셋에서 이미지 텐서 가져오기
        img = img_tensor.squeeze().numpy()  # 채널 차원 제거 및 NumPy 배열로 변환
        axes[i].imshow(img, cmap='gray')  # 이미지 표시

        # 이미지 위에 선분으로 VEC를 표시
        vec = vecs[i]
        axes[i].plot([0, vec[0] * img_tensor.size()[1]], [0, vec[1] * img_tensor.size()[1]], color='red')

        comparison_vec = comparison_vecs[i]
        axes[i].plot([0, comparison_vec[0] * img_tensor.size()[1]], [0, comparison_vec[1] * img_tensor.size()[1]], color='green')

        axes[i].axis('off')  # 축 레이블 제거
    plt.show()


# 모델 초기화
model = CNN()

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()  # 회귀 문제 사용할 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 데이터셋 인스턴스 생성
dataset = PolygonDataset()
batch_size = 128

# 시각화 함수 호출
# visualize_polygon_dataset(dataset.datasets, dataset.vecs, dataset.vecs, num_images=10)

# 데이터 및 라벨 불러오기
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
labels = torch.tensor([dataset.get_vec(i) for i in range(batch_size)])  # FIXME: dataloader 자체에 적용

# 학습
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs = data
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)  # 실제 레이블을 사용하여 손실 계산

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('[%d] loss: %.3f' %  (epoch + 1, running_loss / 100))

print('Finished Training')

# 테스트
result_vecs = []
for test_data in dataset.test_datasets:
    result_vec = model(test_data)
    result_vecs.append((float(result_vec[0][0]), float(result_vec[0][1])))

visualize_polygon_dataset(dataset.test_datasets, result_vecs, dataset.test_vecs, num_images=10)
