import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys
import numpy as np


np.random.seed(3407)
torch.manual_seed(3407)


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, is_train=True):
        self.scaler = StandardScaler()
        dataframe = dataframe.reset_index(drop=True)
        self.labels = dataframe.iloc[:, 2].astype(float).fillna(0)
        features = dataframe.iloc[:, 6:]
        if is_train:
            self.features = self.scaler.fit_transform(features.fillna(0))
        else:
            self.features = self.scaler.transform(features.fillna(0))
        self.features = self.features.reshape(-1, 16381, 1)  

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        return features, label


class BiLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout_prob=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout_prob if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out.squeeze()


def mape(y_true, y_pred):
    y_true, y_pred = y_true.detach().cpu().numpy().flatten(), y_pred.detach().cpu().numpy().flatten()  
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    return 0  


def model_mape(loader, model, criterion):
    model.eval()
    total_loss = 0
    total_mape = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_mape += mape(labels, outputs)
    avg_loss = total_loss / len(loader)
    avg_mape = total_mape / len(loader)
    print(f'Test MSE: {avg_loss:.4f}, Test MAPE: {avg_mape:.2f}%')
    return avg_loss, avg_mape


def l1_regularization(model, lambda_l1):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm


if __name__ == "__main__":
    sys.stdout = open('../Model_Record/Record/BiLSTMIntTrCn.txt', 'w')
    dataframe = pd.read_csv('../Dataset/CnData.csv', low_memory=False)
    dataframe = dataframe.drop('merge_key1', axis=1)
    train_df, test_df = train_test_split(dataframe, test_size=0.2, random_state=3407)
    train_dataset = TimeSeriesDataset(train_df)
    test_dataset = TimeSeriesDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=25, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM(hidden_size=128, num_layers=2, dropout_prob=0.5)
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=0.0001)

    num_epochs = 10
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    lambda_l1 = 0.0005  

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_mape = 0
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            mse_loss = criterion(outputs, labels)
            mape_loss = mape(labels, outputs)
            l1_loss = l1_regularization(model, lambda_l1)
            total_loss = mse_loss + l1_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            total_train_loss += mse_loss.item()
            total_train_mape += mape_loss
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], MSE Loss: {mse_loss.item():.4f}, MAPE: {mape_loss:.2f}%, L1 Loss: {l1_loss.item():.4f}')

        average_train_mse = total_train_loss / len(train_loader)
        average_train_mape = total_train_mape / len(train_loader)
        print(f'Epoch {epoch + 1}, Average Training MSE: {average_train_mse:.4f}, Training MAPE: {average_train_mape:.2f}%')

        val_loss, val_mape = model_mape(test_loader, model, criterion)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), '../Model_Record/Model/BiLSTMIntTrCn.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch + 1} due to no improvement.")
                break

    sys.stdout.close()
    sys.stdout = sys.__stdout__
