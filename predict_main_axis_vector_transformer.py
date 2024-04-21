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
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim, nhead=2), num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)  # Transformer expects shape: (seq_len, batch_size, input_size)
        out = self.transformer(embedded)
        out = self.fc(out[-1])  # Take the output of the last token
        return out


def calculate_main_direction(points):
    polygon = Polygon(points)
    coords = polygon.minimum_rotated_rectangle.exterior.coords
    vecs = [(coords[i + 1][0] - coord[0], coords[i + 1][1] - coord[1]) for i, coord in enumerate(coords[:-1])]
    vec = [vec for vec in vecs if vec[0] >= 0 and vec[1] > 0][0]

    return torch.tensor(vec)


def generate_dataset(start_index, num_samples):
    dataset = []
    labels = []

    for i in range(start_index, num_samples):
        raw_points = BUILDINGS_DATA_JSON["features"][i]["geometry"]["coordinates"][0]

        polygon = Polygon(raw_points)
        polygon_translated = shapely.affinity.translate(polygon, -polygon.centroid.coords[0][0], -polygon.centroid.coords[0][1])

        points = torch.tensor(polygon_translated.exterior.coords)
        dataset.append(points)

        main_direction = calculate_main_direction(points)

        labels.append(main_direction)

    return dataset, labels


def visualize_results(test_cases, predictions):
    num_test_cases = len(test_cases)
    num_cols = 4
    num_rows = (num_test_cases + num_cols - 1) // num_cols

    plt.figure(figsize=(5 * num_cols, 5 * num_rows))

    for i in range(num_test_cases):
        points = test_cases[i]
        pred_direction = predictions[i]

        plt.subplot(num_rows, num_cols, i + 1)
        plt.plot(points[:, 0], points[:, 1], 'b-')
        plt.scatter(points[:, 0], points[:, 1], color='r')

        calculated_direction = calculate_main_direction(points)
        plt.arrow(points[0][0], points[0][1], calculated_direction[0], calculated_direction[1], head_width=0.5, head_length=0.5, fc='r', ec='r')

        plt.arrow(points[0][0], points[0][1], pred_direction[0], pred_direction[1], head_width=0.5, head_length=0.5, fc='g', ec='g')

        plt.title(f"Test Case {i+1}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_samples = 512
    num_tests = 16
    train_dataset, train_labels = generate_dataset(0, num_samples)
    test_cases, _ = generate_dataset(num_samples, num_samples + num_tests)

    input_dim = 2
    hidden_dim = 16
    output_dim = 2

    model = DirectionPredictionModel(input_dim, hidden_dim, output_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 2000
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, train_data in enumerate(train_dataset):
            optimizer.zero_grad()

            output = model(train_data.unsqueeze(0).float())  # Add .float() for compatibility
            label = train_labels[i].unsqueeze(0)

            loss = criterion(output, label.float())  # Add .float() for compatibility
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataset):.4f}')

    predictions = []
    for points in test_cases:
        pred_direction = model(points.unsqueeze(0).float()).detach().numpy()  # Add .float() for compatibility
        predictions.append(pred_direction)

    visualize_results(test_cases, predictions)
