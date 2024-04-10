import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import shapely.affinity
from shapely.geometry import Polygon

with open("data/buildings_data_divided/196164.22754000025_449303.8666800002_196905.28352000023_451480.8424600002.json", "r") as f:
    BUILDINGS_DATA_JSON = json.load(f)


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

    polygon = Polygon(points)
    coords = polygon.minimum_rotated_rectangle.exterior.coords
    vecs = [(coords[i + 1][0] - coord[0], coords[i + 1][1] - coord[1]) for i, coord in enumerate(coords[:-1])]
    vec = [vec for vec in vecs if vec[0] > 0 and vec[1] > 0][0]

    return torch.tensor(vec)


def generate_dataset(start_index, num_samples):
    dataset = []
    labels = []

    for i in range(start_index, num_samples):
        raw_points = BUILDINGS_DATA_JSON["features"][i]["geometry"]["coordinates"][0]

        polygon = Polygon(raw_points)
        # centroid 위치를 향해. polygon data normalize 개념
        polygon_translated = shapely.affinity.translate(polygon, -polygon.centroid.coords[0][0], -polygon.centroid.coords[0][1])

        points = torch.tensor(polygon_translated.exterior.coords)
        dataset.append(points)

        # 폴리곤의 메인 방향성 계산
        main_direction = calculate_main_direction(points)

        # NOTE: 실제 필지의 크기가 서로 다른데, 결과 vector 를 normalize 하는 것이 맞다는 확신을 아직 가지지 못함
        # main_direction_length = torch.norm(main_direction, p=2)
        # main_direction_normalized = main_direction / main_direction_length

        labels.append(main_direction)

    return dataset, labels


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
    num_tests = 16
    max_num_points = 10  # 최대 점 개수 설정
    train_dataset, train_labels = generate_dataset(0, num_samples)
    test_cases, _ = generate_dataset(num_samples, num_tests)

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
