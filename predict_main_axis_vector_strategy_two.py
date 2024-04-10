import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class DirectionPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DirectionPredictionModel, self).__init__()
        self.hidden_dim = hidden_dim
        # 시퀀스 데이터 처리 방안으로 LSTM 테스트.  이후 Transformer 테스트
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM 모델 통과
        out, _ = self.rnn(x)
        # 마지막 시퀀스의 hidden state를 가져와서 fully connected layer 통과
        out = self.fc(out[:, -1, :])
        return out

# 폴리곤의 메인 방향성을 계산하는 함수 - FIXME: obb 의 값을 사용할 예정입니다.
def calculate_main_direction(points):
    return points[-1] - points[0]

# 데이터셋 생성 함수
def generate_dataset(num_samples, max_num_points):
    dataset = []
    labels = []
    for _ in range(num_samples):
        # 랜덤한 점 개수 선택 (3에서 max_num_points 사이)
        num_points = np.random.randint(3, max_num_points + 1)
        # 랜덤한 점 좌표로 폴리곤 생성
        points = torch.rand(num_points, 2) * 10  # 0에서 10 사이의 랜덤 좌표
        dataset.append(points)
        # 폴리곤의 메인 방향성 계산
        main_direction = calculate_main_direction(points)
        labels.append(main_direction)
    return dataset, labels

# 시각화 함수
def visualize_results(test_cases, predictions):
    plt.figure(figsize=(12, 8))
    for i in range(len(test_cases)):
        points = test_cases[i]
        pred_direction = predictions[i]
        plt.plot(points[:, 0], points[:, 1], 'b-')
        plt.scatter(points[:, 0], points[:, 1], color='r')
        plt.arrow(points[0][0], points[0][1], pred_direction[0], pred_direction[1], head_width=0.5, head_length=0.5, fc='g', ec='g')

    plt.title("Predicted Main Directions of Test Cases")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 데이터셋 생성
    num_samples = 128
    max_num_points = 10  # 최대 점 개수 설정
    train_dataset, train_labels = generate_dataset(num_samples, max_num_points)
    test_cases, _ = generate_dataset(16, max_num_points)

    # 모델 초기화
    input_dim = 2  # 입력은 2차원 좌표
    hidden_dim = 16  # LSTM의 hidden state 차원
    output_dim = 2  # 출력은 2차원 벡터 (방향성)

    model = DirectionPredictionModel(input_dim, hidden_dim, output_dim)

    # 손실 함수와 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습
    num_epochs = 1000
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, train_data in enumerate(train_dataset):
            optimizer.zero_grad()

            output = model(train_data.unsqueeze(0))  # 배치 차원을 추가하여 모델 입력 형태로 변환
            label = train_labels[i].unsqueeze(0)

            loss = criterion(output, label)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataset):.4f}')

    # 테스트 케이스 실행
    predictions = []
    for points in test_cases:
        pred_direction = model(points.unsqueeze(0)).detach().numpy()
        predictions.append(pred_direction)

    # 결과 시각화
    visualize_results(test_cases, predictions)
