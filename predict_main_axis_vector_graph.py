import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import shapely.affinity
from shapely.geometry import Polygon, LineString
import debugvisualizer as dv

with open("data/buildings_data_divided/196164.22754000025_449303.8666800002_196905.28352000023_451480.8424600002.json", "r") as f:
    BUILDINGS_DATA_JSON = json.load(f)

def calculate_main_direction(points):
    polygon = Polygon(points)
    coords = polygon.minimum_rotated_rectangle.exterior.coords
    vecs = [(coords[i + 1][0] - coord[0], coords[i + 1][1] - coord[1]) for i, coord in enumerate(coords[:-1])]
    vec = [vec for vec in vecs if vec[0] >= 0 and vec[1] > 0][0]
    return torch.tensor(vec, dtype=torch.float)

def polygon_to_graph(points):
    num_points = points.size(0)
    edge_index = []
    for i in range(num_points - 1):
        edge_index.append([i, i + 1])
    edge_index.append([num_points - 1, 0])  # Closing the loop

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = points.view(-1, 2)  # Node features (2D coordinates)

    return Data(x=x, edge_index=edge_index)

def generate_dataset(start_index, num_samples):
    dataset = []

    for i in range(start_index, num_samples):
        raw_points = BUILDINGS_DATA_JSON["features"][i]["geometry"]["coordinates"][0]
        polygon = Polygon(raw_points)
        polygon_translated = shapely.affinity.translate(polygon, -polygon.centroid.coords[0][0], -polygon.centroid.coords[0][1])

        points = torch.tensor(list(polygon_translated.exterior.coords), dtype=torch.float)

        # Convert polygon to graph representation
        graph_data = polygon_to_graph(points)
        dataset.append(graph_data)

    return dataset

class GraphDirectionPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphDirectionPredictionModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def visualize_results(test_dataset, predictions):
    num_test_cases = len(test_dataset)
    num_cols = 4  # Number of columns for subplots (adjust as needed)
    num_rows = (num_test_cases + num_cols - 1) // num_cols  # Calculate number of rows

    plt.figure(figsize=(5 * num_cols, 5 * num_rows))  # Adjust figure size based on the number of subplots

    for i in range(num_test_cases):
        data = test_dataset[i]
        points = data.x
        pred_direction = predictions[i]

        # Create a subplot for each test case
        plt.subplot(num_rows, num_cols, i + 1)
        plt.plot(points[:, 0], points[:, 1], 'b-')
        plt.scatter(points[:, 0], points[:, 1], color='r')

        # model 이 predict 한 direction plot
        plt.arrow(points[0][0], points[0][1], pred_direction[0], pred_direction[1], head_width=0.5, head_length=0.5, fc='g', ec='g')

        plt.title(f"Test Case {i+1}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

    plt.tight_layout()  # Adjust subplot layout to prevent overlap
    plt.show()

if __name__ == "__main__":
    # 데이터셋 생성
    num_samples = 512
    num_tests = 16
    train_dataset = generate_dataset(0, num_samples)
    test_dataset = generate_dataset(num_samples, num_samples + num_tests)

    # 모델 초기화
    input_dim = 2  # 입력은 2차원 좌표
    hidden_dim = 16  # GNN의 hidden state 차원
    output_dim = 2  # 출력은 2차원 벡터 (방향성)

    model = GraphDirectionPredictionModel(input_dim, hidden_dim, output_dim)

    # 손실 함수와 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for data in train_dataset:
            optimizer.zero_grad()

            output = model(data)
            label = calculate_main_direction(data.x)  # 폴리곤의 주요 방향을 레이블로 사용

            loss = criterion(output, label.unsqueeze(0).float())
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataset):.4f}')

            # 테스트 케이스 실행
            model.eval()
            predictions = []
            for data in test_dataset:
                output = model(data)
                predictions.append(output.detach().numpy())

            # 결과 시각화
            visualize_results(test_dataset, predictions)
