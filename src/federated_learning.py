import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

class DriverClientNode:
    """
    Simulated Local Driver Client Node for Privacy-Preserving Federated Learning.
    Computes local parameter updates without sending raw facial images or video streams to central server.
    """

    def __init__(self, client_id, X_train, y_train, X_val, y_val):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.model = LogisticRegression(
            max_iter=20,
            warm_start=True,
            random_state=42
        )
        # Initialize model
        self.model.fit(X_train[:10], y_train[:10])

    def get_parameters(self):
        """Extract model weight matrix & bias vector"""
        return [self.model.coef_.copy(), self.model.intercept_.copy()]

    def set_parameters(self, weights):
        """Set model weight matrix & bias vector"""
        self.model.coef_ = weights[0].copy()
        self.model.intercept_ = weights[1].copy()

    def fit_local_epoch(self, global_weights):
        """Train local driver model on confidential local dataset"""
        self.set_parameters(global_weights)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(), len(self.X_train)

    def evaluate(self, global_weights):
        """Evaluate global model on local driver validation set"""
        self.set_parameters(global_weights)
        preds = self.model.predict(self.X_val)
        proba = self.model.predict_proba(self.X_val)
        acc = accuracy_score(self.y_val, preds)
        loss = log_loss(self.y_val, proba, labels=[0, 1])
        return loss, acc

def federated_averaging(client_weights_list, sample_sizes):
    """
    FedAvg Algorithm: Computes weighted average of local driver client model weights.
    W_global = sum(n_i * W_i) / sum(n_i)
    """
    total_samples = sum(sample_sizes)
    avg_coef = np.zeros_like(client_weights_list[0][0])
    avg_intercept = np.zeros_like(client_weights_list[0][1])

    for (coef, intercept), n_samples in zip(client_weights_list, sample_sizes):
        weight_factor = n_samples / total_samples
        avg_coef += weight_factor * coef
        avg_intercept += weight_factor * intercept

    return [avg_coef, avg_intercept]

def run_federated_simulation(csv_path, num_clients=3, num_rounds=5):
    """
    Simulates Privacy-Preserving Federated Learning across multiple driver nodes.
    Exchanges only model parameters rather than raw video streams.
    """
    print(f"\nStarting Federated Learning (FedAvg) Simulation across {num_clients} Driver Nodes...")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found. Run feature_extraction.py first.")
        return

    df = pd.read_csv(csv_path, on_bad_lines='skip')
    feature_cols = [col for col in df.columns if col not in ['target', 'filename']]
    
    # Clean data
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df = df.dropna(subset=['target'])
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Shuffle dataset to ensure balanced class distribution across driver nodes
    from sklearn.utils import shuffle
    X_scaled, y = shuffle(X_scaled, y, random_state=42)

    # Distribute data into decentralized driver shards (simulating different vehicles/drivers)
    driver_clients = []
    shard_size = len(X_scaled) // num_clients
    for i in range(num_clients):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_clients - 1 else len(X_scaled)
        Xc, yc = X_scaled[start:end], y[start:end]

        split = int(len(Xc) * 0.8)
        client = DriverClientNode(
            client_id=f"Driver_Node_{i+1}",
            X_train=Xc[:split], y_train=yc[:split],
            X_val=Xc[split:], y_val=yc[split:]
        )
        driver_clients.append(client)


    # Initialize Global Model parameters
    global_weights = driver_clients[0].get_parameters()

    print("\n" + "="*50)
    print("FEDERATED LEARNING (FedAvg) AGGREGATION ROUUNDS")
    print("="*50)

    for r in range(1, num_rounds + 1):
        local_weights_list = []
        sample_sizes = []
        round_losses, round_accs = [], []

        # Local Driver Client Training
        for client in driver_clients:
            weights, n_samples = client.fit_local_epoch(global_weights)
            loss, acc = client.evaluate(weights)
            local_weights_list.append(weights)
            sample_sizes.append(n_samples)
            round_losses.append(loss)
            round_accs.append(acc)

        # Secure Federated Aggregation (FedAvg)
        global_weights = federated_averaging(local_weights_list, sample_sizes)

        mean_acc = np.mean(round_accs)
        mean_loss = np.mean(round_losses)
        print(f"Round {r}/{num_rounds} | Global Model Acc: {mean_acc*100:.2f}% | Loss: {mean_loss:.4f} | Driver Nodes: {num_clients}")

    print("\n[SUCCESS] Federated Learning simulation completed successfully!")
    print("Privacy Preserved: No raw driver facial images or video streams were transmitted.")

if __name__ == "__main__":
    csv_file = r"c:\Users\surya\Downloads\drow\data\extracted_features.csv"
    run_federated_simulation(csv_file)

