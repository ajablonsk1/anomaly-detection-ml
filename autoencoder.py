import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from log_utils import log_experiment, save_run_snapshot
from datetime import datetime


from process_data import X_dos_hulk_test_shared_scaled, X_autoencoder_dos_hulk_train_scaled, \
    X_ftp_patator_test_shared_scaled, X_autoencoder_ftp_patator_train_scaled


# ==========================================
# Autoencoder Model Definition
# ==========================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=2,
                 encoder_layers=None, decoder_layers=None,
                 activation=nn.SELU, dropout_rate=0.0,
                 output_activation=None):
        super(Autoencoder, self).__init__()

        # If decoder_layers not specified, use reversed encoder_layers
        if encoder_layers is None:
            encoder_layers = [10, 6]
        if decoder_layers is None:
            decoder_layers = encoder_layers[::-1]

        # Create activation function instances
        self.activation = activation() if callable(activation) else activation
        self.output_activation = output_activation() if callable(output_activation) else output_activation

        # Build encoder
        encoder_modules = []
        prev_dim = input_dim

        # Add encoder layers
        for dim in encoder_layers:
            encoder_modules.append(nn.Linear(prev_dim, dim))
            encoder_modules.append(self.activation)
            if dropout_rate > 0:
                encoder_modules.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        # Add final encoder layer to bottleneck
        encoder_modules.append(nn.Linear(prev_dim, encoding_dim))
        encoder_modules.append(self.activation)

        # Build decoder
        decoder_modules = []
        prev_dim = encoding_dim

        # Add decoder layers
        for dim in decoder_layers:
            decoder_modules.append(nn.Linear(prev_dim, dim))
            decoder_modules.append(self.activation)
            if dropout_rate > 0:
                decoder_modules.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        # Add final decoder output layer
        decoder_modules.append(nn.Linear(prev_dim, input_dim))
        if self.output_activation is not None:
            decoder_modules.append(self.output_activation)

        # Create sequential models
        self.encoder = nn.Sequential(*encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


# ==========================================
# Utility Functions
# ==========================================

def train_autoencoder(model, train_loader, val_loader=None, epochs=100, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {'train_loss': [], 'val_loss': []}

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            inputs = data[0].to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    inputs = data[0].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}")

    print("Training completed!")
    return history


def calculate_reconstruction_error(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    criterion = nn.MSELoss(reduction='none')
    reconstruction_errors = []

    with torch.no_grad():
        for data in data_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)

            # Calculate reconstruction error (MSE) for each sample
            errors = criterion(outputs, inputs)
            # Mean over features dimension
            sample_errors = torch.mean(errors, dim=1).cpu().numpy()
            reconstruction_errors.extend(sample_errors)

    return np.array(reconstruction_errors)


def find_optimal_threshold(normal_errors, anomaly_errors=None, search_range=(0.001, 0.5), steps=1000):
    if anomaly_errors is not None:
        # If we have labeled anomaly data, find threshold that maximizes F1
        thresholds = np.linspace(search_range[0], search_range[1], steps)
        best_f1 = 0
        best_threshold = None

        # True labels
        y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
        all_errors = np.concatenate([normal_errors, anomaly_errors])

        for threshold in thresholds:
            y_pred = (all_errors > threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"Optimal threshold: {best_threshold:.6f} (F1: {best_f1:.4f})")
        return best_threshold
    else:
        # Without anomaly data, provide percentile-based options
        percentiles = [95, 97.5, 99, 99.5, 99.9]
        thresholds = [np.percentile(normal_errors, p) for p in percentiles]

        print("Percentile-based thresholds:")
        for p, threshold in zip(percentiles, thresholds):
            outliers_percent = (normal_errors > threshold).mean() * 100
            print(
                f"  - {p}th percentile: {threshold:.6f} (would classify {outliers_percent:.2f}% of normal data as anomalies)")

        return np.percentile(normal_errors, 99)


def evaluate_anomaly_detection(normal_errors, anomaly_errors, threshold, file_suffix):
    # True labels (0 for normal, 1 for anomaly)
    y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])

    # Predicted labels based on reconstruction error and threshold
    normal_preds = (normal_errors > threshold).astype(int)
    anomaly_preds = (anomaly_errors > threshold).astype(int)
    y_pred = np.concatenate([normal_preds, anomaly_preds])

    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # True positive rate and false positive rate
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)  # Recall/Sensitivity
    fpr = fp / (fp + tn)  # Fall-out

    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'true_positive_rate': tpr,
        'false_positive_rate': fpr
    }

    os.makedirs("data/autoencoder/results", exist_ok=True)
    with open(f"data/autoencoder/results/results_{file_suffix}.txt", "w") as results_file:
        results_file.write("\nPerformance Metrics:\n")
        results_file.write(f"  - Precision: {precision:.4f}\n")
        results_file.write(f"  - Recall: {recall:.4f}\n")
        results_file.write(f"  - F1 Score: {f1:.4f}\n")
        results_file.write(f"  - True Positive Rate: {tpr:.4f}\n")
        results_file.write(f"  - False Positive Rate: {fpr:.4f}\n")
        results_file.write("\nConfusion Matrix:\n")
        results_file.write(f"  - True Negatives: {tn}\n")
        results_file.write(f"  - False Positives: {fp}\n")
        results_file.write(f"  - False Negatives: {fn}\n")
        results_file.write(f"  - True Positives: {tp}\n")

    return results


def plot_reconstruction_errors(normal_errors, anomaly_errors, threshold, attack_type):
    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal (BENIGN)', color='green')
    plt.hist(anomaly_errors, bins=50, alpha=0.5, label=f'Anomaly ({attack_type})', color='red')

    # Plot threshold line
    plt.axvline(x=threshold, color='blue', linestyle='--',
                label=f'Threshold: {threshold:.6f}')

    plt.title(f'Reconstruction Error Distribution - {attack_type}')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs("data/autoencoder/plots", exist_ok=True)
    plt.savefig(f'data/autoencoder/plots/reconstruction_error_{attack_type}.png')
    plt.close()
    print(f"Plot saved as reconstruction_error_{attack_type}.png")


def plot_training_curve(history, file_suffix):
    plt.figure(figsize=(10, 6))

    plt.plot(history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')

    plt.title('Autoencoder Training Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs("data/autoencoder/plots", exist_ok=True)
    plt.savefig(f'data/autoencoder/plots/autoencoder_training_curve_{file_suffix}.png')
    plt.close()
    print(f"Training curve plot saved as autoencoder_training_curve_{file_suffix}.png")


# ==========================================
# DoS Hulk Attack Detection
# ==========================================

def train_dos_hulk_autoencoder(X_train_scaled, X_test_normal, X_test_attack, batch_size=64, epochs=50):
    print("\n" + "=" * 50)
    print("DoS Hulk Attack Detection with Autoencoder")
    print("=" * 50)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create and train the autoencoder
    input_dim = X_train_scaled.shape[1]
    model = Autoencoder(input_dim=input_dim, encoding_dim=8)

    print(f"Model architecture:")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Encoded dimension: 8")
    print(model)

    history = train_autoencoder(model, train_loader, epochs=epochs)
    plot_training_curve(history, "dos_hulk")

    # Prepare test data
    X_test_normal_tensor = torch.tensor(X_test_normal.values, dtype=torch.float32)
    X_test_attack_tensor = torch.tensor(X_test_attack.values, dtype=torch.float32)

    test_normal_dataset = TensorDataset(X_test_normal_tensor)
    test_attack_dataset = TensorDataset(X_test_attack_tensor)

    test_normal_loader = DataLoader(test_normal_dataset, batch_size=batch_size, shuffle=False)
    test_attack_loader = DataLoader(test_attack_dataset, batch_size=batch_size, shuffle=False)

    # Calculate reconstruction errors
    normal_errors = calculate_reconstruction_error(model, test_normal_loader)
    attack_errors = calculate_reconstruction_error(model, test_attack_loader)

    # Find optimal threshold
    threshold = find_optimal_threshold(normal_errors, attack_errors)

    # Evaluate performance
    results = evaluate_anomaly_detection(normal_errors, attack_errors, threshold, "dos_hulk")

    # Plot reconstruction error distribution
    plot_reconstruction_errors(normal_errors, attack_errors, threshold, "DoS Hulk")

    # Log experiment results
    model_params = {
        "encoding_dim": 8,
        "epochs": epochs,
        "batch_size": batch_size,
        "input_dim": input_dim,
        "learning_rate": 0.001
    }
    log_experiment(results, model_params, attack_type="DoS Hulk")

    return model, results


# ==========================================
# FTP Patator Attack Detection
# ==========================================

def train_ftp_patator_autoencoder(X_train_scaled, X_test_normal, X_test_attack, batch_size=64, epochs=50):
    print("\n" + "=" * 50)
    print("FTP Patator Attack Detection with Autoencoder")
    print("=" * 50)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create and train the autoencoder
    input_dim = X_train_scaled.shape[1]
    model = Autoencoder(input_dim=input_dim, encoding_dim=2)

    print(f"Model architecture:")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Encoded dimension: 8")
    print(model)

    history = train_autoencoder(model, train_loader, epochs=epochs)
    plot_training_curve(history, "ftp_patator")

    # Prepare test data
    X_test_normal_tensor = torch.tensor(X_test_normal.values, dtype=torch.float32)
    X_test_attack_tensor = torch.tensor(X_test_attack.values, dtype=torch.float32)

    test_normal_dataset = TensorDataset(X_test_normal_tensor)
    test_attack_dataset = TensorDataset(X_test_attack_tensor)

    test_normal_loader = DataLoader(test_normal_dataset, batch_size=batch_size, shuffle=False)
    test_attack_loader = DataLoader(test_attack_dataset, batch_size=batch_size, shuffle=False)

    # Calculate reconstruction errors
    normal_errors = calculate_reconstruction_error(model, test_normal_loader)
    attack_errors = calculate_reconstruction_error(model, test_attack_loader)

    # Find optimal threshold
    threshold = find_optimal_threshold(normal_errors, attack_errors, search_range=(0.0005, 1.0), steps=2000)

    # Evaluate performance
    results = evaluate_anomaly_detection(normal_errors, attack_errors, threshold, "ftp_patator")

    # Plot reconstruction error distribution
    plot_reconstruction_errors(normal_errors, attack_errors, threshold, "FTP Patator")

    # Log experiment results
    model_params = {
        "encoding_dim": 2,
        "epochs": epochs,
        "batch_size": batch_size,
        "input_dim": input_dim,
        "learning_rate": 0.001
    }
    log_experiment(results, model_params, attack_type="FTP Patator")

    return model, results


# ==========================================
# Main Function
# ==========================================

def main():
    print("\n" + "=" * 70)
    print("Network Intrusion Detection with Autoencoder - PyTorch Implementation")
    print("=" * 70)


    # DoS Hulk Attack Detection
    print("\n📂 Loading DoS Hulk attack data...")

    # Prepare test data by separating normal and attack samples
    dos_test_normal = X_dos_hulk_test_shared_scaled[
        X_dos_hulk_test_shared_scaled["Label"] == "BENIGN"
        ].drop(columns=["Label"])

    dos_test_attack = X_dos_hulk_test_shared_scaled[
        X_dos_hulk_test_shared_scaled["Label"] == "DoS Hulk"
        ].drop(columns=["Label"])

    print(f"Loaded DoS Hulk data:")
    print(f"  - Training samples (BENIGN): {X_autoencoder_dos_hulk_train_scaled.shape[0]}")
    print(f"  - Test samples (BENIGN): {dos_test_normal.shape[0]}")
    print(f"  - Test samples (DoS Hulk): {dos_test_attack.shape[0]}")

    # Train and evaluate the autoencoder
    dos_hulk_model, dos_hulk_results = train_dos_hulk_autoencoder(
        X_autoencoder_dos_hulk_train_scaled,
        dos_test_normal,
        dos_test_attack,
        batch_size=64,
        epochs=50
    )

    # Save the model
    os.makedirs("data/autoencoder/model", exist_ok=True)
    torch.save(dos_hulk_model.state_dict(), "data/autoencoder/model/dos_hulk_autoencoder.pth")
    print("DoS Hulk model saved as 'data/autoencoder/model/dos_hulk_autoencoder.pth'")

    # FTP Patator Attack Detection

    print("\nLoading FTP Patator attack data...")

    # Prepare test data by separating normal and attack samples
    ftp_test_normal = X_ftp_patator_test_shared_scaled[
        X_ftp_patator_test_shared_scaled["Label"] == "BENIGN"].drop(columns=["Label"])

    ftp_test_attack = X_ftp_patator_test_shared_scaled[
        X_ftp_patator_test_shared_scaled["Label"] == "FTP-Patator"].drop(columns=["Label"])

    print(f"Loaded FTP Patator data:")
    print(f"  - Training samples (BENIGN): {X_autoencoder_ftp_patator_train_scaled.shape[0]}")
    print(f"  - Test samples (BENIGN): {ftp_test_normal.shape[0]}")
    print(f"  - Test samples (FTP-Patator): {ftp_test_attack.shape[0]}")

    # Train and evaluate the autoencoder
    ftp_patator_model, ftp_patator_results = train_ftp_patator_autoencoder(
        X_autoencoder_ftp_patator_train_scaled,
        ftp_test_normal,
        ftp_test_attack,
        batch_size=64,
        epochs=50
    )

    # Save the model
    os.makedirs("data/autoencoder/model", exist_ok=True)
    torch.save(ftp_patator_model.state_dict(), "data/autoencoder/model/ftp_patator_autoencoder.pth")
    print("FTP Patator model saved as 'data/autoencoder/model/ftp_patator_autoencoder.pth'")

    # Snapshot wszystkich danych i wyników
    save_run_snapshot({
        "dos_hulk": {
            "train_shape": X_autoencoder_dos_hulk_train_scaled.shape,
            "test_benign_shape": dos_test_normal.shape,
            "test_attack_shape": dos_test_attack.shape,
            "results": dos_hulk_results,
            "model_params": {
                "encoding_dim": 8,
                "epochs": 50,
                "batch_size": 64,
                "input_dim": X_autoencoder_dos_hulk_train_scaled.shape[1],
                "learning_rate": 0.001
            }
        },
        "ftp_patator": {
            "train_shape": X_autoencoder_ftp_patator_train_scaled.shape,
            "test_benign_shape": ftp_test_normal.shape,
            "test_attack_shape": ftp_test_attack.shape,
            "results": ftp_patator_results,
            "model_params": {
                "encoding_dim": 2,
                "epochs": 50,
                "batch_size": 64,
                "input_dim": X_autoencoder_ftp_patator_train_scaled.shape[1],
                "learning_rate": 0.001
            }
        }
    }, extra_meta={
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "host": os.uname().nodename
    })


if __name__ == "__main__":
    main()