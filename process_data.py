import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# DoS Hulk Attack Data Processing
# ==========================================

# Loading the dataset with DoS Hulk attack samples
dos_hulk_df=pd.read_csv('data/raw/Wednesday-workingHours.pcap_ISCX.csv')

# Selected features for DoS Hulk detection
dos_features = [' Flow Duration', 'Flow Bytes/s', ' Flow Packets/s',
                ' Total Fwd Packets', ' Total Backward Packets',
                ' Fwd Packet Length Mean', ' Bwd Packet Length Mean',
                ' SYN Flag Count', ' ACK Flag Count', ' Destination Port',
                ' Idle Min', 'Idle Mean', 'Init_Win_bytes_forward',
                ' PSH Flag Count', ' Label']

# Selecting only the relevant features from the dataset
df_dos_hulk_selected = dos_hulk_df[dos_features]

# Filtering data for Decision Tree Classifier - only BENIGN and DoS Hulk classes
df_dos_hulk_dtc = df_dos_hulk_selected[df_dos_hulk_selected[" Label"].isin(["BENIGN", "DoS Hulk"])]
X_dtc_dos_hulk = df_dos_hulk_dtc.drop(columns=[" Label"])  # Features
y_dtc_dos_hulk = df_dos_hulk_dtc[" Label"]  # Target variable

# Filtering data for autoencoder - only BENIGN class for anomaly detection
df_dos_hulk_autoencoder = df_dos_hulk_selected[df_dos_hulk_selected[" Label"] == "BENIGN"]
X_autoencoder_dos_hulk = df_dos_hulk_autoencoder.drop(columns=[" Label"])

# Splitting data into training and test sets for the classifier
X_dtc_dos_hulk_train, X_dos_hulk_test_shared, y_dtc_dos_hulk_train, y_dos_hulk_test_shared = train_test_split(
    X_dtc_dos_hulk, y_dtc_dos_hulk, test_size=0.3, random_state=42
)

# Cleaning training data - replacing infinite values with NaN and removing rows with NaN
X_dtc_dos_hulk_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_dtc_dos_hulk_train.dropna(inplace=True)
# Aligning target variable with cleaned features
y_dtc_dos_hulk_train = y_dtc_dos_hulk_train.loc[X_dtc_dos_hulk_train.index].reset_index(drop=True)
X_dtc_dos_hulk_train.reset_index(drop=True, inplace=True)

# Standardizing data - fitting scaler on training data only
scaler_dos_hulk = StandardScaler()
scaler_dos_hulk.fit(X_dtc_dos_hulk_train)

# Scaling training data for classifier
X_dtc_dos_hulk_train_scaled = pd.DataFrame(
    scaler_dos_hulk.transform(X_dtc_dos_hulk_train),
    columns=X_dtc_dos_hulk_train.columns
)
X_dtc_dos_hulk_train_scaled["Label"] = y_dtc_dos_hulk_train  # Adding back the target variable

# Cleaning test data
X_dos_hulk_test_shared.replace([np.inf, -np.inf], np.nan, inplace=True)
X_dos_hulk_test_shared.dropna(inplace=True)
# Aligning target variable with cleaned features
y_dos_hulk_test_shared = y_dos_hulk_test_shared.loc[X_dos_hulk_test_shared.index].reset_index(drop=True)
X_dos_hulk_test_shared.reset_index(drop=True, inplace=True)

# Scaling test data using the same scaler fitted on training data
X_dos_hulk_test_shared_scaled = pd.DataFrame(
    scaler_dos_hulk.transform(X_dos_hulk_test_shared),
    columns=X_dos_hulk_test_shared.columns
)
X_dos_hulk_test_shared_scaled["Label"] = y_dos_hulk_test_shared  # Adding back the target variable

# Processing data for autoencoder - splitting into train/test
X_autoencoder_dos_hulk_train, _ = train_test_split(X_autoencoder_dos_hulk, test_size=0.3, random_state=42)
# Cleaning autoencoder training data
X_autoencoder_dos_hulk_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_autoencoder_dos_hulk_train.dropna(inplace=True)
X_autoencoder_dos_hulk_train.reset_index(drop=True, inplace=True)

# Scaling autoencoder training data using the same scaler as classifier
X_autoencoder_dos_hulk_train_scaled = pd.DataFrame(
    scaler_dos_hulk.transform(X_autoencoder_dos_hulk_train),
    columns=X_autoencoder_dos_hulk_train.columns
)

# ==========================================
# FTP Patator Attack Data Processing
# ==========================================

# Loading the dataset with FTP Patator attack samples
ftp_patator_df=pd.read_csv('data/raw/Tuesday-WorkingHours.pcap_ISCX.csv')

# Selected features for FTP Patator detection
ftp_features = [
    " Flow Duration", " Total Fwd Packets", " Total Backward Packets", "Flow Bytes/s", " Flow Packets/s",
    " Fwd Packet Length Mean", " Fwd Packet Length Std", " Bwd Packet Length Mean", " Bwd Packet Length Std",
    " ACK Flag Count", " RST Flag Count", " Destination Port", " Idle Min", " Idle Max", "Idle Mean", " PSH Flag Count",
    " Label"
]
# Selecting only the relevant features from the dataset
df_ftp_patator_selected = ftp_patator_df[ftp_features]

# Data for classifier - only BENIGN and FTP-Patator classes
df_ftp_patator_dtc = df_ftp_patator_selected[df_ftp_patator_selected[" Label"].isin(["BENIGN", "FTP-Patator"])]
X_dtc_ftp_patator = df_ftp_patator_dtc.drop(columns=[" Label"])  # Features
y_dtc_ftp_patator = df_ftp_patator_dtc[" Label"]  # Target variable

# Data for autoencoder - only BENIGN class for anomaly detection
df_ftp_patator_autoencoder = df_ftp_patator_selected[df_ftp_patator_selected[" Label"] == "BENIGN"]
X_autoencoder_ftp_patator = df_ftp_patator_autoencoder.drop(columns=[" Label"])

# Split classifier data into training and test sets
X_dtc_ftp_patator_train, X_ftp_patator_test_shared, y_dtc_ftp_patator_train, y_ftp_patator_test_shared = train_test_split(
    X_dtc_ftp_patator, y_dtc_ftp_patator, test_size=0.3, random_state=42
)

# Clean training data from infinite values and NaN
X_dtc_ftp_patator_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_dtc_ftp_patator_train.dropna(inplace=True)
# Align target variable with cleaned features
y_dtc_ftp_patator_train = y_dtc_ftp_patator_train.loc[X_dtc_ftp_patator_train.index].reset_index(drop=True)
X_dtc_ftp_patator_train.reset_index(drop=True, inplace=True)

# Scaling with common scaler (fit only on training data)
scaler_ftp_patator = StandardScaler()
scaler_ftp_patator.fit(X_dtc_ftp_patator_train)

# Scale training data for classifier
X_dtc_ftp_patator_train_scaled = pd.DataFrame(
    scaler_ftp_patator.transform(X_dtc_ftp_patator_train),
    columns=X_dtc_ftp_patator_train.columns
)
X_dtc_ftp_patator_train_scaled["Label"] = y_dtc_ftp_patator_train  # Adding back the target variable

# Clean and scale test data
X_ftp_patator_test_shared.replace([np.inf, -np.inf], np.nan, inplace=True)
X_ftp_patator_test_shared.dropna(inplace=True)
# Align target variable with cleaned features
y_ftp_patator_test_shared = y_ftp_patator_test_shared.loc[X_ftp_patator_test_shared.index].reset_index(drop=True)
X_ftp_patator_test_shared.reset_index(drop=True, inplace=True)

# Scale test data using the same scaler as for training data
X_ftp_patator_test_shared_scaled = pd.DataFrame(
    scaler_ftp_patator.transform(X_ftp_patator_test_shared),
    columns=X_ftp_patator_test_shared.columns
)
X_ftp_patator_test_shared_scaled["Label"] = y_ftp_patator_test_shared  # Adding back the target variable

# Process data for autoencoder - split into train/test
X_autoencoder_ftp_patator_train, _ = train_test_split(X_autoencoder_ftp_patator, test_size=0.3, random_state=42)
# Clean autoencoder training data
X_autoencoder_ftp_patator_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_autoencoder_ftp_patator_train.dropna(inplace=True)
X_autoencoder_ftp_patator_train.reset_index(drop=True, inplace=True)

# Scale autoencoder training data using the same scaler as classifier
X_autoencoder_ftp_patator_train_scaled = pd.DataFrame(
    scaler_ftp_patator.transform(X_autoencoder_ftp_patator_train),
    columns=X_autoencoder_ftp_patator_train.columns
)

# ==========================================
# Test Functions for Data Verification
# ==========================================

def check_missing_or_inf(df, name):
    """Check for missing or infinite values in the dataframe"""
    nan_count = df.isna().sum().sum()
    inf_count = df.isin([np.inf, -np.inf]).sum().sum()
    print(f"✅ {name} - missing values: {nan_count}, infinite values: {inf_count}")

def show_distribution_stats(df, name):
    """Show basic statistics about the data distribution"""
    print(f"\n📊 Distribution statistics: {name}")
    print(df.drop(columns=["Label"], errors="ignore").describe().loc[["mean", "std"]].T)

def print_shapes(*dfs):
    """Print shapes of multiple dataframes"""
    print("\n📐 Data shapes:")
    for df in dfs:
        print(f"{df[0]}: {df[1].shape}")

def show_labels(*dfs):
    """Show unique label values in dataframes"""
    print("\n🏷️ Unique labels:")
    for df in dfs:
        if "Label" in df[1].columns:
            print(f"{df[0]}: {df[1]['Label'].unique()}")

def preview(df, name):
    """Show sample data from dataframe"""
    print(f"\n🔍 {name} - sample data:")
    print(df.head(3))


# ==========================================
# Testing DoS Hulk Data
# ==========================================

check_missing_or_inf(X_dtc_dos_hulk_train_scaled, "X_dtc_dos_hulk_train_scaled")
check_missing_or_inf(X_dos_hulk_test_shared_scaled, "X_dos_hulk_test_shared_scaled")
check_missing_or_inf(X_autoencoder_dos_hulk_train_scaled, "X_autoencoder_dos_hulk_train_scaled")

show_distribution_stats(X_dtc_dos_hulk_train_scaled, "X_dtc_dos_hulk_train_scaled")

print_shapes(
    ("X_dtc_dos_hulk_train_scaled", X_dtc_dos_hulk_train_scaled),
    ("X_dos_hulk_test_shared_scaled", X_dos_hulk_test_shared_scaled),
    ("X_autoencoder_dos_hulk_train_scaled", X_autoencoder_dos_hulk_train_scaled)
)

show_labels(
    ("X_dtc_dos_hulk_train_scaled", X_dtc_dos_hulk_train_scaled),
    ("X_dos_hulk_test_shared_scaled", X_dos_hulk_test_shared_scaled)
)

preview(X_dtc_dos_hulk_train_scaled, "X_dtc_dos_hulk_train_scaled")

# ==========================================
# Testing FTP Patator Data
# ==========================================

check_missing_or_inf(X_dtc_ftp_patator_train_scaled, "X_dtc_ftp_patator_train_scaled")
check_missing_or_inf(X_ftp_patator_test_shared_scaled, "X_ftp_patator_test_shared_scaled")
check_missing_or_inf(X_autoencoder_ftp_patator_train_scaled, "X_autoencoder_ftp_patator_train_scaled")

show_distribution_stats(X_dtc_ftp_patator_train_scaled, "X_dtc_ftp_patator_train_scaled")

print_shapes(
    ("X_dtc_ftp_patator_train_scaled", X_dtc_ftp_patator_train_scaled),
    ("X_ftp_patator_test_shared_scaled", X_ftp_patator_test_shared_scaled),
    ("X_autoencoder_ftp_patator_train_scaled", X_autoencoder_ftp_patator_train_scaled)
)

show_labels(
    ("X_dtc_ftp_patator_train_scaled", X_dtc_ftp_patator_train_scaled),
    ("X_ftp_patator_test_shared_scaled", X_ftp_patator_test_shared_scaled)
)

preview(X_dtc_ftp_patator_train_scaled, "X_dtc_ftp_patator_train_scaled")