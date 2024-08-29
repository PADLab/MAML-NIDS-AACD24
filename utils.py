import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
import numpy as np
import os
import warnings
import time
import seaborn as sns
from statics import (
    BENIGN_MALWARE_RATIO,
    HIDDEN_SIZE,
    DEVICE,
    EPOCHS,
    META_EPOCHS,
    LEARNING_RATE,
)

warnings.filterwarnings("ignore")

print(f"The device is: {DEVICE}")


class UnswDataset(Dataset):
    def __init__(self, df, transform=None, device=DEVICE):
        super().__init__()

        self.df = df
        self.transform = transform

        self.x = torch.tensor(
            self.df.iloc[:, :-1].values, dtype=torch.float32, device=device
        )
        self.y = torch.tensor(
            self.df.iloc[:, -1].values, dtype=torch.float32, device=device
        )
        self.y = self.y.unsqueeze(1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def get_model(model_filepath, input_size=79, hidden_size=128, output_size=12):
    model = Net(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_filepath))
    return model


def maml_update(model, loss, lr):
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    updated_params = {
        name: param - lr * grad
        for (name, param), grad in zip(model.named_parameters(), grads)
    }
    return updated_params


def apply_updated_params(model, input_size, hidden_size, output_size, updated_params):
    adapted_model = Net(input_size, hidden_size, output_size).to(DEVICE)
    state_dict = model.state_dict()

    for name, param in state_dict.items():
        if name in updated_params:
            state_dict[name].copy_(updated_params[name])

    adapted_model.load_state_dict(state_dict)
    return adapted_model


def provide_data_balance_before_binary(
    df: pd.DataFrame, TASK_NAME: str
) -> pd.DataFrame:
    # Calculate final sample size
    malware_size = df[df["Label"] != "Benign"].shape[0]
    benign_size = df[df["Label"] == "Benign"].shape[0]
    malware_to_keep = benign_size * (1 / BENIGN_MALWARE_RATIO) - benign_size
    task_sample_size = df[df["Label"] == TASK_NAME].shape[0]

    malware_removal_ratio = 1 - malware_to_keep / (malware_size - task_sample_size)

    # Remove malware samples from each attack type
    for attack_type in df["Label"].unique():
        if attack_type == "Benign":
            continue
        attack_type_size = df[df["Label"] == attack_type].shape[0]
        remove_samples = int(attack_type_size * malware_removal_ratio)
        df.drop(
            df[df["Label"] == attack_type]
            .sample(n=remove_samples, random_state=42)
            .index,
            inplace=True,
        )

    return df


def train(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    Net: nn.Module,
    epochs: int = 30,
    output_filepath: str = "./Models",
):
    # Hyperparameters
    input_size = len(df.columns) - 1
    hidden_size = HIDDEN_SIZE
    output_size = 1
    batch_size = 64
    lr = LEARNING_RATE

    # Model, loss and optimizer
    model = Net(input_size, hidden_size, output_size).to(DEVICE)
    criterion = nn.BCELoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Data loaders
    train_dataset = UnswDataset(df)
    test_dataset = UnswDataset(df_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    train_losses = []
    test_losses = []
    accuracies = []
    f1_scores = []
    t1 = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0
        accuracy = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        all_y = []
        all_y_pred = []

        with torch.no_grad():
            for x, y in test_loader:
                y_pred = model(x)
                loss = criterion(y_pred, y)

                accuracy += ((y_pred > 0.5) == y).sum().item() / len(y)

                test_loss += loss.item()

                y_pred_binary = (y_pred > 0.5).int()
                y_binary = y.int()

                true_positives += ((y_pred_binary == 1) & (y_binary == 1)).sum().item()
                false_positives += ((y_pred_binary == 1) & (y_binary == 0)).sum().item()
                false_negatives += ((y_pred_binary == 0) & (y_binary == 1)).sum().item()

                all_y.extend(y_binary.cpu().numpy())
                all_y_pred.extend(y_pred_binary.cpu().numpy())

        test_loss /= len(test_loader)
        accuracy /= len(test_loader)

        f1 = f1_score(all_y, all_y_pred)

        test_losses.append(test_loss)
        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(
            f"""Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, \
        Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}"""
        )
    t2 = time.time()

    print(f"Average epoch duration: {(t2 - t1) / EPOCHS:2f}")
    if not os.path.exists(os.path.join(output_filepath, "model.pth")):
        torch.save(model.state_dict(), os.path.join(output_filepath, "model.pth"))
        print("Model saved.")

    return model, train_losses, test_losses, accuracies, f1_scores


def meta_train(
    df: pd.DataFrame,
    df_test: pd.DataFrame,
    df_task: pd.DataFrame,
    epochs=30,
    output_filepath="./Models",
):
    input_size = len(df.columns) - 1
    hidden_size = HIDDEN_SIZE
    output_size = 1
    task_lr = LEARNING_RATE * 5
    meta_lr = LEARNING_RATE
    meta_batch_size = 32

    meta_model = Net(input_size, hidden_size, output_size).to(DEVICE)
    meta_model.load_state_dict(torch.load(os.path.join(output_filepath, "model.pth")))
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=meta_lr)
    criterion = nn.BCELoss().to(DEVICE)

    # Drop attacks from df_train the number of samples in df_task
    size_to_drop = len(df_task)
    if size_to_drop > len(df[df["Label"] == 1]):
        print("Size to drop is greater than the number of malware samples.")
        print(
            f"Size to drop: {size_to_drop}, \
Malware samples: {len(df[df['Label'] == 1])}"
        )
        print("Dropping 10% of malware samples.")
        print(f"Data distribution before dropping: {df['Label'].value_counts()}")
        size_to_drop = len(df[df["Label"] == 1]) // 10
        df_task = df_task.sample(n=size_to_drop, random_state=42)
        print(f"Data distribution after dropping: {df['Label'].value_counts()}")

    dropped_df = df.drop(
        df[df["Label"] == 1].sample(n=size_to_drop, random_state=42).index
    )
    dropped_df.reset_index(drop=True, inplace=True)

    # Combine df_task with df_train
    meta_df: pd.DataFrame = pd.concat([df_task, dropped_df])
    meta_df.reset_index(drop=True, inplace=True)

    train_dataset = UnswDataset(meta_df)
    test_dataset = UnswDataset(df_test)

    train_loader = DataLoader(train_dataset, batch_size=meta_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=meta_batch_size, shuffle=False)

    # Meta training loop
    losses = []
    meta_losses = []
    test_losses = []
    accuracies = []
    f1_scores = []

    t1 = time.time()
    for meta_epoch in range(epochs):
        meta_model.train()

        train_loss = 0
        meta_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            meta_optimizer.zero_grad()

            y_pred = meta_model(x)
            loss = criterion(y_pred, y)
            train_loss += loss.item()

            updated_params = maml_update(meta_model, loss, task_lr)

            adapted_model = apply_updated_params(
                meta_model, input_size, hidden_size, output_size, updated_params
            )

            y_pred = adapted_model(x)
            loss2 = criterion(y_pred, y)

            meta_loss += loss2

        meta_loss /= len(train_loader)
        meta_loss.backward()
        meta_optimizer.step()

        meta_losses.append(meta_loss.item())

        train_loss /= len(train_loader)
        losses.append(train_loss)

        meta_model.eval()

        accuracy = 0
        test_loss = 0

        all_y = []
        all_y_pred = []

        with torch.no_grad():
            for x, y in test_loader:
                y_pred = meta_model(x)
                loss = criterion(y_pred, y)
                test_loss += loss.item()
                accuracy += ((y_pred > 0.5) == y).sum().item() / len(y)

                y_pred_binary = (y_pred > 0.5).int()
                y_binary = y.int()

                all_y.extend(y_binary.cpu().numpy())
                all_y_pred.extend(y_pred_binary.cpu().numpy())

        test_loss /= len(test_loader)
        accuracy /= len(test_loader)

        f1 = f1_score(all_y, all_y_pred)

        test_losses.append(test_loss)
        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(
            f"""Meta Epoch {meta_epoch+1}/{META_EPOCHS}, \
        Train Loss: {train_loss:.4f}, Meta Loss: {meta_loss:.4f}, \
        Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}"""
        )

    t2 = time.time()
    print(f"Average epoch duration: {(t2 - t1) / EPOCHS:2f}")
    # Save the meta model
    if not os.path.exists(os.path.join(output_filepath, "meta_model.pth")):
        torch.save(
            meta_model.state_dict(), os.path.join(output_filepath, "meta_model.pth")
        )
        print("Meta Model saved.")

    return meta_model, losses, meta_losses, test_losses, accuracies, f1_scores


def confusion_matrix_calc(model, df_test, matrix_names, binary=True):
    y_true_multiclass = (
        df_test["True_Label"].apply(lambda x: matrix_names.index(x)).values
    )
    y_pred_binary = []

    dataset = UnswDataset(df_test.iloc[:, :-1], device=DEVICE)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            if binary:
                y_pred_binary.extend((model(x) > 0.5).int().cpu().numpy())
            else:
                y_pred_binary.extend(torch.argmax(model(x), dim=1).cpu().numpy())

    y_pred_multiclass = []
    for i, binary_pred in enumerate(y_pred_binary):
        if binary_pred == 0:
            y_pred_multiclass.append(
                matrix_names.index("Benign")
            )  # Replace with actual benign class
        else:
            y_pred_multiclass.append(
                y_true_multiclass[i]
            )  # Replace with actual malicious class

    cm = np.zeros((2, len(matrix_names)), dtype=int)

    for i in range(len(y_true_multiclass)):
        true_class = y_true_multiclass[i]
        predicted_binary = y_pred_binary[i]
        cm[predicted_binary, true_class] += 1

    return cm


def plot_confusion_matrix(cm, classes, output_name, TASK_NAME):
    if cm.shape[0] == 2:
        y_classes = ["Benign", "Malicious"]
    else:
        y_classes = classes

    plt.figure(figsize=(11, 11))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"shrink": 0.35},
        xticklabels=classes,
        yticklabels=y_classes,
        square=True,
    )
    plt.title(f"Confusion Matrix for {TASK_NAME}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(output_name)
    plt.close()


def generate_confusion_matrices(
    df_test: pd.DataFrame,
    model: nn.Module,
    plot_filepath: str,
    TASK_NAME: str,
    classes: list,
):

    cm = confusion_matrix_calc(model, df_test, classes)

    plot_confusion_matrix(cm, classes, plot_filepath, TASK_NAME)


def preprocess_data(
    df: pd.DataFrame,
    TASK_NAME: str,
    benign_label="Benign",
):
    columns_to_drop = [
        "Flow ID",
        "Timestamp",
        "Src IP",
        "Dst IP",
        "Connection Type",
    ]

    df.drop(columns=columns_to_drop, inplace=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_task = df[df["Label"] == TASK_NAME]

    df.drop(df_task.index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = provide_data_balance_before_binary(df, TASK_NAME)

    scaler = StandardScaler()

    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    df_task.iloc[:, :-1] = scaler.transform(df_task.iloc[:, :-1])

    df_test_benign = df[df["Label"] == benign_label].sample(frac=0.2, random_state=42)
    df_test_malware = df[df["Label"] != benign_label].sample(frac=0.2, random_state=42)

    df_test = pd.concat([df_test_benign, df_test_malware])
    del df_test_benign, df_test_malware

    # Save the data labels before going binary to create a confusion matrix
    df_labels = df["Label"].copy()
    df_test_labels = df_test["Label"].copy()
    df_task_labels = df_task["Label"].copy()

    df["Label"] = df["Label"].apply(lambda x: 0 if x == benign_label else 1)
    df_test["Label"] = df_test["Label"].apply(lambda x: 0 if x == benign_label else 1)
    df_task["Label"] = df_task["Label"].apply(lambda _: 1)

    df["True_Label"] = df_labels
    df_test["True_Label"] = df_test_labels
    df_task["True_Label"] = df_task_labels

    df.drop(df_test.index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df, df_test, df_task


def plot_results(
    train_losses: list,
    test_losses: list,
    accuracies: list,
    f1_scores: list,
    outuput_filename: str = "./Plots/",
    meta_losses: list | None = None,
):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].plot(train_losses, label="Train Loss")
    if meta_losses:
        ax[0].plot(meta_losses, label="Meta Loss")
    ax[0].plot(test_losses, label="Test Loss")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("EPOCHS")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(accuracies, label="Accuracy")
    ax[1].plot(f1_scores, label="F1 Score")
    ax[1].set_title("F1 Score and Accuracy")
    ax[1].set_xlabel("EPOCHS")
    ax[1].set_ylabel("Score")
    ax[1].legend()

    if meta_losses and not os.path.exists(
        os.path.join(outuput_filename, "meta_training_plot.png")
    ):
        plt.savefig(os.path.join(outuput_filename, "meta_training_plot.png"))
    elif not os.path.exists(os.path.join(outuput_filename, "training_plot.png")):
        plt.savefig(os.path.join(outuput_filename, "training_plot.png"))
