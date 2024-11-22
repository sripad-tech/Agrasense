#Author: Sripad Madhusudan Upadhyaya

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Set the device to CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the same Dual-Channel 1D CNN Model
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

# Function to prepare data for inference
def prepare_inference_data(sample):
    # sample shape: (seq_len, 2)
    input_tensor = torch.tensor(sample, dtype=torch.float32).transpose(0, 1)  # Shape: (2, seq_len)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_tensor

# Function to scale data
def scale_inference_data(input_tensor, scaler):
    batch_size, channels, seq_len = input_tensor.shape

    # Flatten for scaling (batch_size * seq_len, channels)
    flat_input = input_tensor.permute(0, 2, 1).contiguous().view(-1, channels).cpu().numpy()
    flat_input_scaled = scaler.transform(flat_input)

    # Reshape back to (batch_size, channels, seq_len)
    input_scaled = torch.tensor(flat_input_scaled).view(batch_size, seq_len, channels).permute(0, 2, 1)
    return input_scaled

# Inference function
def infer_nitrate_turbidity(model, scaler, sample):
    model.eval()
    with torch.no_grad():
        input_tensor = prepare_inference_data(sample).to(device)
        input_scaled = scale_inference_data(input_tensor, scaler).to(device)
        nitrate_pred, turbidity_pred = model(input_scaled)
        nitrate_pred = nitrate_pred.cpu().numpy().squeeze()
        turbidity_pred = turbidity_pred.cpu().numpy().squeeze()
    return nitrate_pred, turbidity_pred

# Main function for inference
if __name__ == "__main__":
    # Load the trained model and scaler
    model = DualChannel1DCNN(num_filters=64, kernel_size=5).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    scaler = joblib.load('scaler.pkl')

    # Load your sample data for inference
    # The sample data should be a numpy array of shape (seq_len, 2)
    # where seq_len is the number of data points, and 2 corresponds to [Wavelength, Absorbance]

    # Example: Load sample data from an Excel file
    # Replace 'new_sample_data.xlsx' with your actual data file
    sample_data_df = pd.read_excel('new_sample_data.xlsx')
    sample_data_df.replace(',', '.', regex=True, inplace=True)

    # Ensure that the dataframe has columns named 'Wavelength' and 'Absorbance'
    sample_data = sample_data_df[['Wavelength', 'Absorbance']].values

    # Sort the data by wavelength if necessary
    # sample_data = sample_data[np.argsort(sample_data[:, 0])]

    # Infer nitrate concentration and turbidity
    nitrate_prediction, turbidity_prediction = infer_nitrate_turbidity(model, scaler, sample_data)

    print(f"Predicted Nitrate Concentration: {nitrate_prediction}")
    print(f"Predicted Turbidity: {turbidity_prediction}")
