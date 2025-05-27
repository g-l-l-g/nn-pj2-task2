# loss_landscape_experiment.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import freeze_support
from datetime import datetime

from VGG_BatchNorm.data_loader.loaders import get_cifar_loader
from VGG_BatchNorm.models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters
from VGG_BatchNorm.train import train_and_collect_batch_losses
from VGG_BatchNorm.utils.plotting import plot_loss_landscape


# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置
ROOT_DATA_DIR = './data_cifar'
ROOT_SAVE_DIR_LANDSCAPE = f'./runs/loss_landscape_study/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
LOG_DIR_TB_BATCH = os.path.join(ROOT_SAVE_DIR_LANDSCAPE, 'tensorboard_batch_logs')
SAVE_PATH_PLOT_LANDSCAPE = os.path.join(ROOT_SAVE_DIR_LANDSCAPE, 'images')

# 损失状况研究的训练参数
NUM_EPOCHS_LANDSCAPE = 20
BATCH_SIZE_LANDSCAPE = 128
N_ITEMS_LANDSCAPE = -1  # 设置为-1，表示使用数据集全部数据
NUM_WORKERS_LANDSCAPE = 4

DEFAULT_OPTIMIZER_CLASS = optim.SGD
DEFAULT_OPTIMIZER_PARAMS = {
    "momentum": 0.9,
}

# 用于损失状况测试的学习率
LEARNING_RATES_LANDSCAPE = [1e-3, 2e-3, 1e-4, 5e-4]

# 损失状况研究的模型配置
model_configs_landscape = [
    {
        "plot_name": "Standard VGG",
        "tb_name_prefix": "VGG_A",
        "model_class": VGG_A,
        "params": {"num_classes": 10, "init_weights_flag": True}
    },
    {
        "plot_name": "Standard VGG + BatchNorm",
        "tb_name_prefix": "VGG_A_BN_Full",
        "model_class": VGG_A_BatchNorm,
        "params": {"num_classes": 10, "init_weights_flag": True, "batch_norm_2d": True, "batch_norm_1d": True}
    }
]


def run_landscape_study():
    print(f"使用设备: {DEVICE}")
    os.makedirs(ROOT_DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR_TB_BATCH, exist_ok=True)
    os.makedirs(SAVE_PATH_PLOT_LANDSCAPE, exist_ok=True)

    print("为损失状况研究加载 CIFAR-10 数据集...")
    try:
        from torchvision import datasets
        datasets.CIFAR10(root=ROOT_DATA_DIR, train=True, download=True)
        datasets.CIFAR10(root=ROOT_DATA_DIR, train=False, download=True)  # 此脚本不使用验证集，但下载是个好习惯
        print("CIFAR-10 数据检查/下载完成。")
    except Exception as e:
        print(f"警告: CIFAR-10 数据下载/检查失败: {e}")

    train_loader_landscape = get_cifar_loader(
        root=ROOT_DATA_DIR, batch_size=BATCH_SIZE_LANDSCAPE, train=True, shuffle=True,  # shuffle 很重要
        num_workers=NUM_WORKERS_LANDSCAPE, n_items=N_ITEMS_LANDSCAPE
    )

    print(
        f"损失状况的训练加载器: {len(train_loader_landscape.dataset)} 个样本, 每个 epoch {len(train_loader_landscape)} 个批次。")
    print(f"每次学习率运行的总预期步数: {len(train_loader_landscape) * NUM_EPOCHS_LANDSCAPE}")

    all_runs_batch_losses = {}  # 存储格式: { "Standard VGG": {lr1_str: [损失], lr2_str: [损失]}, ... }

    criterion = nn.CrossEntropyLoss()

    for model_config in model_configs_landscape:
        model_plot_name = model_config["plot_name"]
        model_tb_prefix = model_config["tb_name_prefix"]
        model_class = model_config["model_class"]
        model_params = model_config["params"]

        print(f"\n--- 开始模型运行: {model_plot_name} ---")
        current_model_losses_by_lr = {}  # 当前模型，按学习率存储损失

        for lr in LEARNING_RATES_LANDSCAPE:
            print(f"  训练 {model_plot_name} 使用学习率: {lr} ---")

            # 每次运行重新初始化模型，以确保权重和优化器状态是全新的
            model = model_class(**model_params).to(DEVICE)

            # --- Optimizer Configuration ---
            optimizer_class = model_config.get("optimizer_class", DEFAULT_OPTIMIZER_CLASS)
            current_optimizer_params = DEFAULT_OPTIMIZER_PARAMS.copy()
            current_optimizer_params.update(model_config.get("optimizer_params", {}))
            optimizer = optimizer_class(model.parameters(), lr=lr, **current_optimizer_params)
            print(f"Using optimizer: {optimizer_class.__name__} with params: {current_optimizer_params}")

            # optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用当前学习率

            model_run_name_tb = f"{model_tb_prefix}_lr_{lr}"

            batch_losses = train_and_collect_batch_losses(
                model_run_name=model_run_name_tb,
                model=model,
                train_loader=train_loader_landscape,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=NUM_EPOCHS_LANDSCAPE,
                device=DEVICE,
                base_log_dir=LOG_DIR_TB_BATCH  # 传递基础日志目录
            )
            # 绘图函数期望字典的键是字符串，例如 'lr_0.001'
            current_model_losses_by_lr[f"lr_{lr}"] = batch_losses
            print(f"  完成 {model_plot_name} 的学习率 {lr}。收集到 {len(batch_losses)} 个批次损失。")

        all_runs_batch_losses[model_plot_name] = current_model_losses_by_lr

    # --- 绘制损失状况图 ---
    print("\n--- 绘制损失状况图 ---")

    vgg_a_losses = all_runs_batch_losses.get("Standard VGG", {})
    vgg_a_bn_losses = all_runs_batch_losses.get("Standard VGG + BatchNorm", {})

    if not vgg_a_losses and not vgg_a_bn_losses:  # 检查是否有数据可绘图
        print("未收集到用于绘图的损失数据。正在退出。")
        return

    plot_loss_landscape(
        loss_data_dict_vgga=vgg_a_losses,
        loss_data_dict_vggbn=vgg_a_bn_losses,
        model_name_vgga="Standard VGG",
        model_name_vggbn="Standard VGG + BatchNorm",  # 与示例图中的图例匹配
        save_path_base=SAVE_PATH_PLOT_LANDSCAPE  # 图像保存的基础路径
    )
    print("损失状况图生成尝试完成。")
    print(f"请参考 {os.path.join(SAVE_PATH_PLOT_LANDSCAPE, 'loss_landscape_comparison.png')}")
    print(f"要查看 TensorBoard 批次日志，请运行: tensorboard --logdir {LOG_DIR_TB_BATCH}")


if __name__ == '__main__':
    freeze_support()
    run_landscape_study()
