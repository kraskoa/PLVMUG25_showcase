import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv("daily-website-visitors.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.rename(columns={"Unique.Visits": "Unique_Visits"}, inplace=True)
df["Unique_Visits"] = df["Unique_Visits"].str.replace(",", "").astype(int)
df = df.set_index("Date")
visitors = df["Unique_Visits"].values.astype(float).reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(-1, 1))
visitors_scaled = scaler.fit_transform(visitors)


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


SEQ_LENGTH = 30
X, y = create_sequences(visitors_scaled, SEQ_LENGTH)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions


model = LSTMForecaster()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for i in range(epochs):
    for seq, labels in train_loader:
        optimizer.zero_grad()
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()
    if i % 5 == 0:
        print(f"Epoch {i} loss: {single_loss.item():.4f}")

model.eval()
test_predictions = []
for i in range(len(X_test_tensor)):
    seq = X_test_tensor[i : i + 1]
    with torch.no_grad():
        test_predictions.append(model(seq).item())

predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
actuals = scaler.inverse_transform(y_test)

plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size + SEQ_LENGTH :], actuals, label="Actual Visitors")
plt.plot(
    df.index[train_size + SEQ_LENGTH :],
    predictions,
    label="Forecasted Visitors",
    linestyle="--",
)
plt.title("Website Visitor Forecast vs Actuals")
plt.xlabel("Date")
plt.ylabel("Number of Unique Visitors")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("forecast_vs_actuals.png")
