import torch

ENUM = {'Benign': 0, 
        'ICMP Flood': 1, 
        'Slowloris': 2, 
        'SYN Flood': 3, 
        'UDP Flood': 4, 
        'DNS Flood': 5, 
        'Dictionary Attack': 6, 
        'OS Scan': 7, 
        'Port Scan': 8, 
        'Ping Sweep': 9, 
        'Vulnerability Scan': 10, 
        'ARP Spoofing': 11}

BENIGN_MALWARE_RATIO = 0.67
EPOCHS = 30
META_EPOCHS = 30
HIDDEN_SIZE = 16
LEARNING_RATE = 0.002
OUTPUT_PATH = "./Runs"
DATA_PATH = "./ACI-IoT-2023.csv"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'