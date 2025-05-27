# bn_comparison.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
from multiprocessing import freeze_support
from datetime import datetime

# 导入自定义模块
from VGG_BatchNorm.data_loader.loaders import get_cifar_loader
from VGG_BatchNorm.models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters
from VGG_BatchNorm.train import train_single_model_experiment
from VGG_BatchNorm.utils.plotting import plot_single_experiment_results

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# 路径配置
ROOT_DATA_DIR = './data_cifar'
ROOT_SAVE_DIR = f'./runs/bn_comparison_experiments/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
LOG_DIR_BASE_TENSORBOARD = os.path.join(ROOT_SAVE_DIR, 'tensorboard_logs')
SAVE_PATH_PLOTS_BASE = os.path.join(ROOT_SAVE_DIR, 'images')
SAVED_MODELS_DIR_BASE = os.path.join(ROOT_SAVE_DIR, 'saved_models')

# 训练参数
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 30
N_ITEMS_TRAIN = -1  # Use 1000 for faster testing, -1 for full dataset
N_ITEMS_VAL = -1  # Use 200 for faster testing, -1 for full dataset
NUM_WORKERS = 2

# --- 默认优化器和调度器配置 ---
DEFAULT_OPTIMIZER_CLASS = optim.Adam
DEFAULT_OPTIMIZER_PARAMS = {"lr": LEARNING_RATE}
DEFAULT_SCHEDULER_CLASS = optim.lr_scheduler.MultiStepLR
DEFAULT_SCHEDULER_PARAMS = {
    "milestones": [20, 27],
    "gamma": 0.1
}

# --- 模型配置 ---
model_configurations = [
    {
        "name": "VGG_A",
        "model_class": VGG_A,
        "params": {"num_classes": 10, "init_weights_flag": True}
    },
    {
        "name": "VGG_A_BN_2d_1d",
        "model_class": VGG_A_BatchNorm,
        "params": {"num_classes": 10, "init_weights_flag": True, "batch_norm_2d": True, "batch_norm_1d": True}
    },
    {
        "name": "VGG_A_BN_2d_only",
        "model_class": VGG_A_BatchNorm,
        "params": {"num_classes": 10, "init_weights_flag": True, "batch_norm_2d": True, "batch_norm_1d": False}
    },
    {
        "name": "VGG_A_BN_1d_only",
        "model_class": VGG_A_BatchNorm,
        "params": {"num_classes": 10, "init_weights_flag": True, "batch_norm_2d": False, "batch_norm_1d": True}
    },
]


def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"Set seed to {seed_value}")


def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    os.makedirs(ROOT_DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR_BASE_TENSORBOARD, exist_ok=True)
    os.makedirs(SAVE_PATH_PLOTS_BASE, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR_BASE, exist_ok=True)

    print("Loading CIFAR-10 dataset...")
    train_loader = get_cifar_loader(
        root=ROOT_DATA_DIR, batch_size=BATCH_SIZE, train=True, shuffle=True,
        num_workers=NUM_WORKERS, n_items=N_ITEMS_TRAIN
    )
    val_loader = get_cifar_loader(
        root=ROOT_DATA_DIR, batch_size=BATCH_SIZE, train=False, shuffle=False,
        num_workers=NUM_WORKERS, n_items=N_ITEMS_VAL
    )
    print(f"Train loader: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Validation loader: {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    all_experiment_results = {}
    criterion = nn.CrossEntropyLoss()

    for config in model_configurations:
        model_name = config["name"]
        model_class = config["model_class"]
        model_params = config["params"]

        print(f"\n--- Experiment: {model_name} ---")
        model = model_class(**model_params).to(DEVICE)
        print(f"Number of parameters for {model_name}: {get_number_of_parameters(model)}")

        # --- Optimizer Configuration ---
        optimizer_class = config.get("optimizer_class", DEFAULT_OPTIMIZER_CLASS)

        # Start with default optimizer params, then update with model-specific params
        current_optimizer_params = DEFAULT_OPTIMIZER_PARAMS.copy()
        current_optimizer_params.update(config.get("optimizer_params", {}))

        optimizer = optimizer_class(model.parameters(), **current_optimizer_params)
        print(f"Using optimizer: {optimizer_class.__name__} with params: {current_optimizer_params}")

        # --- Scheduler Configuration ---
        scheduler = None
        scheduler_class = config.get("scheduler_class", DEFAULT_SCHEDULER_CLASS)
        current_scheduler_params = None
        if scheduler_class:
            current_scheduler_params = DEFAULT_SCHEDULER_PARAMS.copy()
            current_scheduler_params.update(config.get("scheduler_params", {}))
            scheduler = scheduler_class(optimizer, **current_scheduler_params)
            print(f"Using scheduler: {scheduler_class.__name__} with params: {current_scheduler_params}")

        # --- Hyperparameters for Logging ---
        current_hparams = {
            "model_name_variant": model_name,
            "batch_size": BATCH_SIZE,
            "optimizer": optimizer_class.__name__,
            **current_optimizer_params  # Add all actual optimizer params used
        }
        if scheduler:
            current_hparams["scheduler"] = scheduler_class.__name__
            for k_s, v_s in current_scheduler_params.items():
                current_hparams[f"scheduler_{k_s}"] = v_s

        # Directory for saving this model's weights
        model_weights_save_dir = os.path.join(SAVED_MODELS_DIR_BASE, model_name)
        os.makedirs(model_weights_save_dir, exist_ok=True)

        results = train_single_model_experiment(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            base_log_dir=LOG_DIR_BASE_TENSORBOARD,
            hparams_dict=current_hparams,
            scheduler=scheduler,
            model_save_dir=model_weights_save_dir
        )
        all_experiment_results[model_name] = results
        print(f"Finished training {model_name}")
        plot_single_experiment_results(model_name, results, SAVE_PATH_PLOTS_BASE, NUM_EPOCHS)

    # --- 5. 可视化总比较 ---
    print("\n--- Plotting Overall Comparison Results ---")
    epochs_range = range(1, NUM_EPOCHS + 1)
    # Ensure enough colors if many models are compared
    num_models = len(all_experiment_results)
    colors = plt.cm.get_cmap('tab10', num_models).colors if num_models > 0 else []
    if num_models == 0:  # Handle case with no results
        print("No experiment results to plot.")
        return

    linestyles_map = {"train_losses": ":", "val_losses": "-", "train_accuracies": ":", "val_accuracies": "-"}

    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)  # Loss comparison
    for i, (model_name, results) in enumerate(all_experiment_results.items()):
        color = colors[i % len(colors)]
        plt.plot(
            epochs_range,
            results["train_losses"],
            label=f'{model_name} Train Loss',
            color=color,
            linestyle=linestyles_map["train_losses"]
        )
        plt.plot(
            epochs_range,
            results["val_losses"],
            label=f'{model_name} Val Loss',
            color=color,
            linestyle=linestyles_map["val_losses"]
        )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Overall Loss Comparison')
    plt.legend(fontsize='small', loc='upper right')
    plt.grid(True)

    plt.subplot(1, 2, 2)  # Accuracy comparison
    for i, (model_name, results) in enumerate(all_experiment_results.items()):
        color = colors[i % len(colors)]
        plt.plot(
            epochs_range,
            results["train_accuracies"],
            label=f'{model_name} Train Acc',
            color=color,
            linestyle=linestyles_map["train_accuracies"]
        )
        plt.plot(
            epochs_range,
            results["val_accuracies"],
            label=f'{model_name} Val Acc',
            color=color,
            linestyle=linestyles_map["val_accuracies"]
        )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Overall Accuracy Comparison')
    plt.legend(fontsize='small', loc='lower right')
    plt.grid(True)

    plt.tight_layout()
    overall_comparison_plot_path = os.path.join(SAVE_PATH_PLOTS_BASE, "overall_training_summary_comparison.png")
    plt.savefig(overall_comparison_plot_path)
    plt.close()
    print(f"Overall comparison plot saved to {overall_comparison_plot_path}")

    print("\n--- Experiment Complete ---")
    print(f"To view TensorBoard logs, run: tensorboard --logdir {LOG_DIR_BASE_TENSORBOARD}")
    print(f"Saved models (best weights) are in: {SAVED_MODELS_DIR_BASE}")


if __name__ == '__main__':
    freeze_support()
    main()
