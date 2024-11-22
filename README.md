# Agrasense
# Author : Sripad Madhusudan Upadhyaya

# **Dual-Channel 1D CNN for Nitrate Concentration and Turbidity Prediction**

## **Table of Contents**

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Inference](#inference)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## **Introduction**

This project implements a **Dual-Channel 1D Convolutional Neural Network (CNN)** to predict **nitrate concentration** and **turbidity** from spectral data (wavelength and absorbance). The model processes two input channels—wavelength and absorbance—using separate CNN layers and combines them to make predictions.

---

## **Project Structure**

```
project/
│
├── data/
│   └── augmented_dataset.xlsx      # Training dataset
│
├── models/
│   └── best_model.pth              # Saved PyTorch model
│   └── scaler.pkl                  # Saved StandardScaler
│
├── outputs/
│   └── loss_curve.png              # Training and validation loss plot
│   └── actual_vs_predicted_nitrate.png  # Prediction results plot
│
├── src/
│   └── train.py                    # Training script
│   └── inference.py                # Inference script
│
├── README.md                       # Project README
└── requirements.txt                # Required Python packages
```

---

## **Prerequisites**

- **Python 3.6 or higher**
- **PyTorch 1.7 or higher**
- **Additional Python packages:** Listed in `requirements.txt`

---

## **Installation**

1. **Clone the Repository**

   ```bash
   git clone https://gitlab.com/your_username/your_repository.git
   cd your_repository
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

   **Contents of `requirements.txt`:**

   ```
   torch
   torchvision
   pandas
   numpy
   scikit-learn
   matplotlib
   joblib
   openpyxl
   ```

---

## **Usage**

### **1. Data Preparation**

- Place your dataset file (e.g., `augmented_dataset.xlsx`) in the `data/` directory.
- Ensure the dataset is properly formatted. See [Data Preparation](#data-preparation) for details.

### **2. Training the Model**

- Navigate to the `src/` directory:

  ```bash
  cd src/
  ```

- Run the training script:

  ```bash
  python train.py
  ```

### **3. Inference**

- After training, use the inference script to make predictions on new data:

  ```bash
  python inference.py
  ```

---

## **Data Preparation**

The model expects an Excel file (`.xlsx`) with the following columns:

1. **Wavelength**: The wavelength values.
2. **Absorbance**: The corresponding absorbance readings.
3. **Nitrate Concentration**: The target nitrate concentration for the sample.
4. **Turbidity**: The target turbidity value for the sample.

**Data Format:**

- The data should be organized such that each sample's spectral data is listed sequentially.
- A change in the sequence of wavelength values (from high to low) indicates a new sample.

**Example:**

| Wavelength | Absorbance | Nitrate Concentration | Turbidity |
|------------|------------|-----------------------|-----------|
| 200        | 0.123      | 10                    | 5         |
| 201        | 0.130      | 10                    | 5         |
| ...        | ...        | ...                   | ...       |
| 200        | 0.115      | 15                    | 7         |
| 201        | 0.120      | 15                    | 7         |
| ...        | ...        | ...                   | ...       |

---

## **Model Architecture**

### **Dual-Channel 1D CNN**

- **Input:** Two channels (wavelength and absorbance).
- **CNN Layers:**
  - Separate convolutional layers for each input channel.
  - Convolutional layers extract features from the input sequences.
- **Feature Combination:**
  - Flatten and concatenate features from both channels.
- **Fully Connected Layers:**
  - First, predict turbidity from combined features.
  - Then, concatenate turbidity prediction with features to predict nitrate concentration.

**Model Diagram:**

```
            ┌───────────────────┐
            │   Input Data      │
            │ (2, sequence_len) │
            └─────────┬─────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
┌─────────▼────────┐     ┌────────▼────────┐
│ Wavelength CNN   │     │ Absorbance CNN  │
└─────────┬────────┘     └────────┬────────┘
          │                       │
          └───────────┬───────────┘
                      │
            ┌─────────▼─────────┐
            │  Concatenate      │
            └─────────┬─────────┘
                      │
                 ┌────▼────┐
                 │ Flatten │
                 └────┬────┘
                      │
               ┌──────▼──────┐
               │ Fully Conn. │
               │  (Turbidity)│
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │ Concatenate │
               │  (Features +│
               │  Turbidity) │
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │ Fully Conn. │
               │ (Nitrate)   │
               └─────────────┘
```

---

## **Training the Model**

### **Script: `train.py`**

The training script performs the following steps:

1. **Data Loading and Preprocessing:**
   - Loads the dataset from the specified Excel file.
   - Preprocesses the data to extract samples, nitrate concentrations, and turbidity values.
   - Converts data into PyTorch tensors.

2. **Data Splitting:**
   - Splits the data into training and validation sets (80% training, 20% validation).

3. **Data Scaling:**
   - Scales the input data using `StandardScaler`.
   - Saves the scaler for use during inference.

4. **Model Initialization:**
   - Initializes the `DualChannel1DCNN` model.
   - Defines the loss function (`MSELoss`) and optimizer (`Adam`).

5. **Training Loop:**
   - Trains the model over a specified number of epochs.
   - Implements early stopping based on validation loss.
   - Saves the best model during training.

6. **Evaluation:**
   - Evaluates the model on the validation set.
   - Calculates the R² score for nitrate concentration prediction.
   - Generates plots for training/validation loss and actual vs. predicted nitrate concentrations.

### **Running the Training Script**

- Execute the script:

  ```bash
  python train.py
  ```

- **Parameters:**
  - `train_data_file`: Path to the training data file.
  - `num_epochs`: Number of training epochs (default: 1000).
  - `early_stop_patience`: Patience for early stopping (default: 150).

- **Output Files:**
  - `best_model.pth`: Saved model weights.
  - `scaler.pkl`: Saved `StandardScaler` instance.
  - `loss_curve.png`: Training and validation loss plot.
  - `actual_vs_predicted_nitrate.png`: Plot comparing actual and predicted nitrate concentrations.

---

## **Evaluating the Model**

- **R² Score:** The script calculates and displays the R² score on the validation set to assess model performance.

- **Visualization:**
  - **Loss Curve:** Shows the training and validation loss over epochs.
  - **Actual vs. Predicted Plot:** Visualizes how well the model's predictions match the actual nitrate concentrations.

---

## **Inference**

### **Script: `inference.py`**

The inference script allows you to use the trained model to make predictions on new data.

1. **Load the Model and Scaler:**
   - Loads the saved model (`best_model.pth`) and scaler (`scaler.pkl`).

2. **Prepare New Data:**
   - Loads new data from an Excel file.
   - Preprocesses and scales the data using the saved scaler.

3. **Make Predictions:**
   - Uses the model to predict nitrate concentration and turbidity.

4. **Display Results:**
   - Outputs the predicted values.

### **Running the Inference Script**

- Execute the script:

  ```bash
  python inference.py
  ```

- **Parameters:**
  - Ensure that the path to the new data file is correctly specified in the script.

---

## **Results**

- **Expected Outcomes:**
  - The model should predict nitrate concentration and turbidity with reasonable accuracy.
  - The R² score and plots provide insights into the model's performance.

- **Improving Performance:**
  - Adjust hyperparameters (e.g., number of filters, kernel size, learning rate).
  - Experiment with different network architectures or additional preprocessing steps.

---

## **Troubleshooting**

- **Common Issues:**

  - **CUDA Errors:**
    - If you encounter CUDA-related errors, ensure that your PyTorch installation matches your CUDA version.
    - Alternatively, run the script on the CPU by setting `device = torch.device('cpu')`.

  - **Data Format Errors:**
    - Ensure your data is correctly formatted and free of inconsistencies.
    - Check for missing values or incorrect data types.

  - **Module Import Errors:**
    - Verify that all required packages are installed.
    - Use `pip install -r requirements.txt` to install missing dependencies.

- **Debugging Tips:**

  - **Print Statements:**
    - Insert print statements to display tensor shapes and values during execution.

  - **Exception Handling:**
    - Wrap code blocks with `try-except` statements to catch and handle exceptions gracefully.

---

## **References**

- **PyTorch Documentation:** [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- **scikit-learn Documentation:** [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- **Matplotlib Documentation:** [https://matplotlib.org/stable/index.html](https://matplotlib.org/stable/index.html)
- **Pandas Documentation:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**

- **Contributors:** List of contributors or team members.
- **Data Sources:** Acknowledge any data sources used in the project.
- **Inspirations:** Mention any resources or projects that inspired this work.

---

## **Contact**

For any questions or suggestions, please contact:

- **Name:** Sripad Madhusudan Upadhyaya
- **Email:** sripadmu7@gmail.com

---

**Note:** This README provides a comprehensive overview of the project, including instructions on how to set up the environment, run the code, and interpret the results.
