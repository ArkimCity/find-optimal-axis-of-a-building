import cv2
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import shapely.affinity
from shapely.geometry import Polygon

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import debugvisualizer as dv

np.random.seed(12)

# 가상의 데이터셋 생성
class PolygonDataset(Dataset):
    def __init__(self, num_samples=1024, num_vertices=6, img_size=32):
        self.num_samples = num_samples
        self.num_test_samples = 128
        self.num_vertices = num_vertices
        self.img_size = img_size

        with open("data/buildings_data_divided/196164.22754000025_449303.8666800002_196905.28352000023_451480.8424600002.json", "r") as f:
            self.buildings_data_json = json.load(f)

        self.datasets, self.vecs = self.make_datasets(0, self.num_samples)
        self.test_datasets, self.test_vecs = self.make_datasets(self.num_samples, self.num_samples + self.num_test_samples)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.datasets[idx]

    def get_vec(self, idx):
        return self.vecs[idx]

    def normalize_coordinates(self, vertices):
        # 다각형의 좌표를 [0, 1] 범위로 정규화
        vertices = np.array(vertices)
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        normalized_vertices = (vertices - min_coords) / (max_coords - min_coords)

        # 이미지 크기에 맞게 좌표 조정 (32x32 이미지 기준)
        scaled_vertices = normalized_vertices * (self.img_size - 1)

        return scaled_vertices

    def make_each_dataset(self, vertices_raw):

        vertices = self.normalize_coordinates(vertices_raw)

        # 결과에 해당할 것으로 예상되는 벡터 - label 로 사용
        polygon = Polygon(vertices)
        polygon_translated = shapely.affinity.translate(polygon, -polygon.bounds[0] + 0.1, -polygon.bounds[1] + 0.1)

        vertices = np.array(polygon_translated.exterior.coords)

        coords = polygon_translated.minimum_rotated_rectangle.exterior.coords
        vecs = [(coords[i + 1][0] - coord[0], coords[i + 1][1] - coord[1]) for i, coord in enumerate(coords[:-1])]
        vec_raw = [vec for vec in vecs if vec[0] >= 0 and vec[1] > 0][0]
        vec = vec_raw
        # vec_length = np.linalg.norm(vec_raw)
        # vec = tuple(float(x / vec_length) for x in vec_raw)

        # 이미지 생성
        img = np.zeros((self.img_size, self.img_size))
        for i in range(len(vertices) - 1):
            pt1 = tuple((vertices[i]).astype(int))
            pt2 = tuple((vertices[i + 1]).astype(int))
            cv2.line(img, pt1, pt2, color=1, thickness=1)

        # PyTorch Tensor로 변환
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0)  # 채널 차원 추가
        return img_tensor, vec

    def make_datasets(self, start_index, end_index):
        img_tensors, vecs = [], []
        for i in range(start_index, end_index):
            vertices = self.buildings_data_json["features"][i]["geometry"]["coordinates"][0]
            img_tensor, vec = self.make_each_dataset(vertices)

            img_tensors.append(img_tensor)
            vecs.append(vec)

        return img_tensors, vecs

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
        axes[i].plot([0, vec[0]], [0, vec[1]], color='red')

        comparison_vec = comparison_vecs[i]
        axes[i].plot([0, comparison_vec[0]], [0, comparison_vec[1]], color='green')

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
all_labels = [
    torch.tensor([dataset.get_vec(i * batch_size + j) for j in range(batch_size)]) for i in range(8)
]  # FIXME: dataloader 자체에 적용

# 학습
num_epochs = 1000
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs = data
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, all_labels[i])  # 실제 레이블을 사용하여 손실 계산

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch+1) % 100 == 0:
        print('[%d] loss: %.3f' %  (epoch + 1, running_loss / 100))

print('Finished Training')

# 테스트
result_vecs = []
for test_data in dataset.test_datasets:
    result_vec = model(test_data.unsqueeze(0))
    result_vecs.append((float(result_vec[0][0]), float(result_vec[0][1])))

visualize_polygon_dataset(dataset.test_datasets, result_vecs, dataset.test_vecs, num_images=10)
