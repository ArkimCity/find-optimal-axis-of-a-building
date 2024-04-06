import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 데이터셋 1: 10개의 점과 1개의 결과 벡터
X1 = torch.rand(10, 2)  # 10개의 2차원 점
y1 = torch.rand(1, 2)   # 1개의 결과 벡터

# 데이터셋 2: 5개의 점과 1개의 결과 벡터
X2 = torch.rand(5, 2)   # 5개의 2차원 점
y2 = torch.rand(1, 2)   # 1개의 결과 벡터

# 모델 정의: 간단한 다층 퍼셉트론 (MLP)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # 입력 차원: 2, 은닉층 뉴런 수: 16
        self.fc2 = nn.Linear(16, 8)  # 은닉층 뉴런 수: 16, 출력 차원: 8
        self.fc3 = nn.Linear(8, 2)   # 출력 차원: 2 (결과 벡터의 차원)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP()  # 모델 인스턴스 생성

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()  # 평균 제곱 오차 손실
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저

# 데이터셋 1에 대한 학습
def train_model(X, y):
    model.train()  # 모델을 학습 모드로 설정
    optimizer.zero_grad()  # 기울기 초기화

    outputs = model(X)  # 모델에 입력을 전달하여 출력 계산
    loss = criterion(outputs, y)  # 출력과 실제 값 사이의 손실 계산
    loss.backward()  # 역전파 수행
    optimizer.step()  # 옵티마이저로 모델 파라미터 업데이트

    return loss.item()  # 현재 배치의 손실 반환

# 데이터셋 1에 대해 학습 수행
for epoch in range(1000):  # 에폭 수 설정
    loss = train_model(X1, y1)
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# 데이터셋 1을 사용하여 테스트 예측 수행
test_points_1 = torch.rand(3, 2)  # 3개의 2차원 점 생성
predicted_y_1 = model(test_points_1)
print("Predictions for dataset 1:")
print(predicted_y_1)

# 데이터셋 2에 대한 학습 수행
for epoch in range(1000):  # 에폭 수 설정
    loss = train_model(X2, y2)
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# 데이터셋 2를 사용하여 테스트 예측 수행
test_points_2 = torch.rand(2, 2)  # 2개의 2차원 점 생성
predicted_y_2 = model(test_points_2)
print("Predictions for dataset 2:")
print(predicted_y_2)
