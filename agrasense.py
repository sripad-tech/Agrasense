#Author : Sripad Madhusudan Upadhyaya

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import os

# Set the device to CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dual-Channel 1D CNN Model
class DualChannel1DCNN(nn.Module):
    def __init__(self, num_filters, kernel_size):
        super(DualChannel1DCNN, self).__init__()

        # CNN layers for wavelength channel
        self.conv1_wavelength = nn.Conv1d(
            in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.conv2_wavelength = nn.Conv1d(
            in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        # CNN layers for absorbance channel
        self.conv1_absorbance = nn.Conv1d(
            in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2_absorbance = nn.Conv1d(
            in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2)

        # Fully connected layers will be initialized dynamically
        self.fc_turbidity_1 = None
        self.fc_turbidity_2 = None
        self.fc_nitrate_1 = None
        self.fc_nitrate_2 = None

    def forward(self, x):
        # x shape: (batch_size, 2, seq_len)
        # Separate channels
        wavelength = x[:, 0, :].unsqueeze(1)  # Shape: (batch_size, 1, seq_len)
        absorbance = x[:, 1, :].unsqueeze(1)  # Shape: (batch_size, 1, seq_len)

        # CNN for wavelength
        out_wavelength = self.relu(self.conv1_wavelength(wavelength))
        out_wavelength = self.maxpool(self.relu(self.conv2_wavelength(out_wavelength)))

        # CNN for absorbance
        out_absorbance = self.relu(self.conv1_absorbance(absorbance))
        out_absorbance = self.maxpool(self.relu(self.conv2_absorbance(out_absorbance)))

        # Flatten and concatenate
        out_wavelength = out_wavelength.view(out_wavelength.size(0), -1)
        out_absorbance = out_absorbance.view(out_absorbance.size(0), -1)
        combined_out = torch.cat((out_wavelength, out_absorbance), dim=1)  # Shape: (batch_size, feature_size)

        # Initialize fully connected layers if not already done
        if self.fc_turbidity_1 is None:
            self.flatten_size = combined_out.size(1)
            self.fc_turbidity_1 = nn.Linear(self.flatten_size, 64).to(device)
            self.fc_turbidity_2 = nn.Linear(64, 1).to(device)
            self.fc_nitrate_1 = nn.Linear(64 + 1, 32).to(device)
            self.fc_nitrate_2 = nn.Linear(32, 1).to(device)

        # Turbidity prediction
        turbidity_hidden = self.relu(self.fc_turbidity_1(combined_out))
        turbidity_pred = self.fc_turbidity_2(turbidity_hidden)

        # Concatenate turbidity prediction
        nitrate_input = torch.cat((turbidity_hidden, turbidity_pred), dim=1)

        # Nitrate prediction
        nitrate_hidden = self.relu(self.fc_nitrate_1(nitrate_input))
        nitrate_pred = self.fc_nitrate_2(nitrate_hidden)

        return nitrate_pred, turbidity_pred

# Data Preparation Functions
def load_training_data(file_path):
    df = pd.read_excel(file_path)
    df.replace(',', '.', regex=True, inplace=True)
    return preprocess_data(df)

def preprocess_data(df):
    samples = []
    nitrate_concentration = []
    turbidity = []
    current_sample = []
    prev_wavelength = df.iloc[0, 0]
    prev_nitrate = df.iloc[0, 2]
    prev_turbidity = df.iloc[0, 3]

    for index, row in df.iterrows():
        wavelength = float(row.iloc[0])
        absorbance = float(row.iloc[1])
        if wavelength < prev_wavelength:  # New sample
            if current_sample:
                samples.append(current_sample)
                nitrate_concentration.append(float(prev_nitrate))
                turbidity.append(float(prev_turbidity))
                current_sample = []
        current_sample.append([wavelength, absorbance])
        prev_wavelength = wavelength
        prev_nitrate = row.iloc[2]
        prev_turbidity = row.iloc[3]

    if current_sample:
        samples.append(current_sample)
        nitrate_concentration.append(float(prev_nitrate))
        turbidity.append(float(prev_turbidity))

    # Convert to numpy arrays
    samples = [np.array(sample) for sample in samples]
    nitrate_concentration = np.array(nitrate_concentration)
    turbidity = np.array(turbidity)

    return samples, nitrate_concentration, turbidity

def prepare_data(samples, nitrate_concentration, turbidity):
    inputs = []
    nitrate_targets = []
    turbidity_values = []

    for sample, nitrate, turb in zip(samples, nitrate_concentration, turbidity):
        sample = np.array(sample)
        inputs.append(torch.tensor(sample, dtype=torch.float32).transpose(0, 1))  # Shape (2, seq_len)
        nitrate_targets.append(torch.tensor(nitrate, dtype=torch.float32))
        turbidity_values.append(torch.tensor(turb, dtype=torch.float32))

    inputs = torch.stack(inputs).to(device)
    nitrate_targets = torch.stack(nitrate_targets).to(device)
    turbidity_values = torch.stack(turbidity_values).to(device)

    return inputs, nitrate_targets, turbidity_values

# Scaling Function
def scale_data(inputs, scaler=None):
    batch_size, channels, seq_len = inputs.shape

    # Flatten for scaling (batch_size * seq_len, channels)
    flat_inputs = inputs.permute(0, 2, 1).contiguous().view(-1, channels).cpu().numpy()

    if scaler is None:
        scaler = StandardScaler()
        flat_inputs_scaled = scaler.fit_transform(flat_inputs)
    else:
        flat_inputs_scaled = scaler.transform(flat_inputs)

    # Reshape back to (batch_size, channels, seq_len)
    inputs_scaled = torch.tensor(flat_inputs_scaled).view(batch_size, seq_len, channels).permute(0, 2, 1).to(device)

    return inputs_scaled, scaler

# Training Function
def train_dual_channel_cnn(train_data_file, num_epochs=1000, early_stop_patience=150):
    samples, nitrate_concentration, turbidity = load_training_data(train_data_file)
    inputs, nitrate_targets, turbidity_values = prepare_data(samples, nitrate_concentration, turbidity)

    train_inputs, val_inputs, train_nitrate, val_nitrate, train_turbidity, val_turbidity = train_test_split(
        inputs, nitrate_targets, turbidity_values, test_size=0.2, random_state=42
    )

    # Move data to device
    train_inputs = train_inputs.to(device)
    train_nitrate = train_nitrate.to(device)
    train_turbidity = train_turbidity.to(device)
    val_inputs = val_inputs.to(device)
    val_nitrate = val_nitrate.to(device)
    val_turbidity = val_turbidity.to(device)

    model = DualChannel1DCNN(num_filters=64, kernel_size=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    early_stop_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        inputs_scaled, scaler = scale_data(train_inputs)
        nitrate_pred, turbidity_pred = model(inputs_scaled)
        loss_nitrate = criterion(nitrate_pred.squeeze(), train_nitrate)
        loss_turbidity = criterion(turbidity_pred.squeeze(), train_turbidity)
        total_loss = loss_nitrate + loss_turbidity  # Multi-task loss
        total_loss.backward()
        optimizer.step()

        # Validation
        val_inputs_scaled, _ = scale_data(val_inputs, scaler)
        model.eval()
        with torch.no_grad():
            val_nitrate_pred, val_turbidity_pred = model(val_inputs_scaled)
            loss_nitrate_val = criterion(val_nitrate_pred.squeeze(), val_nitrate)
            loss_turbidity_val = criterion(val_turbidity_pred.squeeze(), val_turbidity)
            val_loss = loss_nitrate_val + loss_turbidity_val

        # Record losses
        train_losses.append(total_loss.item())
        val_losses.append(val_loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Training Loss: {total_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Plot Losses
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

    # Load the best model for generating predictions
    model.load_state_dict(torch.load('best_model.pth', map_location=device))

    # Generate predictions on validation set
    val_inputs_scaled, _ = scale_data(val_inputs, scaler)
    model.eval()
    with torch.no_grad():
        val_nitrate_pred, _ = model(val_inputs_scaled)

    # Convert the predicted and actual nitrate concentrations to numpy arrays
    val_nitrate_pred = val_nitrate_pred.cpu().numpy().squeeze()
    actual_nitrate = val_nitrate.cpu().numpy().squeeze()

    # Calculate R^2 Score
    r2 = r2_score(actual_nitrate, val_nitrate_pred)
    print(f"Validation R^2 Score: {r2:.4f}")

    # Plot actual vs predicted nitrate concentrations
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_nitrate, val_nitrate_pred, label='Predicted Nitrate', color='blue')
    plt.plot([min(actual_nitrate), max(actual_nitrate)], [min(actual_nitrate), max(actual_nitrate)], 'r--', label='Perfect Fit')
    plt.xlabel('Actual Nitrate Concentration')
    plt.ylabel('Predicted Nitrate Concentration')
    plt.title('Actual vs Predicted Nitrate Concentration')
    plt.legend()
    plt.savefig('actual_vs_predicted_nitrate.png')
    plt.show()

    return model, scaler

# Main function to train the model
if __name__ == "__main__":
    # Define the training data file and number of epochs
    train_data_file = 'augmented_dataset.xlsx'
    num_epochs = 10000

    # Train the model
    model, scaler = train_dual_channel_cnn(train_data_file, num_epochs=num_epochs)
