import pandas as pd
import os
from datetime import datetime
import time
import json

from utils import (
    preprocess_data,
    Net,
    train,
    meta_train,
    plot_results,
    generate_confusion_matrices,
)

from statics import (
    EPOCHS,
    META_EPOCHS,
    OUTPUT_PATH,
    HIDDEN_SIZE,
    LEARNING_RATE,
    DATA_PATH,
    ENUM,
)

if __name__ == "__main__":
    t1 = time.time()

    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(
            "Dataset not found. Please download the",
            "dataset from the link provided in the README.",
        )
        exit(1)

    
    
    date = datetime.now().strftime("%Y-%m-%d-%H:%M")
    out_filepath = os.path.join(OUTPUT_PATH, date)
    os.makedirs(out_filepath, exist_ok=True)

    hyperparameters = {
        "HIDDEN_SIZE": HIDDEN_SIZE,
        "EPOCHS": EPOCHS,
        "META_EPOCHS": META_EPOCHS,
        "LEARNING RATE": LEARNING_RATE,
        "MODULES": str(next(Net(79, HIDDEN_SIZE, 1).modules())),
    }

    with open(os.path.join(out_filepath, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=4)

    task_names = df["Label"].unique().tolist()
    task_names.remove("Benign")

    for task_name in task_names:
        task_path = os.path.join(out_filepath, task_name)
        os.mkdir(task_path)

        if os.path.exists(os.path.join(task_path, "model.pth")):
            print(f"Model for {task_name} already exists.")
            continue

        print(f"Training model for {task_name}.")

        df = pd.read_csv(DATA_PATH)

        task_sample_size = df[df["Label"] == task_name].shape[0]
        total_samples = df.shape[0]

        if task_sample_size / total_samples < 0.005:
            print("Task sample size is less than 0.5% of the total samples. Skipping..")
            continue

        # print(
        #     "Percentage of values before preprocessing:",
        #     f"{df['Label'].value_counts(normalize=True) * 100}",
        # )

        df, df_test, df_task = preprocess_data(df, task_name)

        # print(
        #     "Percentage of values after preprocessing:",
        #     f"{df['Label'].value_counts(normalize=True) * 100}",
        # )

        if not os.path.exists(f"Models/{task_name}_model.pth"):
            # train the model without the column "True_Label"
            model, train_losses, test_losses, accuracies, f1_scores = train(
                df.drop(columns=["True_Label"]),
                df_test.drop(columns=["True_Label"]),
                Net,
                EPOCHS,
                task_path,
            )

            plot_results(train_losses, test_losses, accuracies, f1_scores, task_path)

            generate_confusion_matrices(
                pd.concat([df_test, df_task]),
                model,
                os.path.join(task_path, "confusion_matrix.png"),
                task_name,
                list(ENUM.keys()),
            )
        else:
            print("Model already exists for training. Skipping..")

        if not os.path.exists(f"Models/{task_name}_meta_model.pth"):
            (
                meta_model,
                train_losses,
                meta_losses,
                test_losses,
                accuracies,
                f1_scores,
            ) = meta_train(
                df.drop(columns=["True_Label"]),
                df_test.drop(columns=["True_Label"]),
                df_task.drop(columns=["True_Label"]),
                META_EPOCHS,
                task_path,
            )

            # Plot the results starting from the old results

            plot_results(
                train_losses,
                test_losses,
                accuracies,
                f1_scores,
                task_path,
                meta_losses,
            )

            generate_confusion_matrices(
                pd.concat([df_test, df_task]),
                meta_model,
                os.path.join(task_path, "meta_confusion_matrix.png"),
                task_name,
                list(ENUM.keys()),
            )
        else:
            print("Meta model already exists. Skipping..")

    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds.")