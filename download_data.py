import kagglehub
import shutil
import os

path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")

os.makedirs("data/raw", exist_ok=True)

shutil.copy(f"{path}/Tuesday-WorkingHours.pcap_ISCX.csv", "data/raw")
shutil.copy(f"{path}/Wednesday-workingHours.pcap_ISCX.csv", "data/raw")