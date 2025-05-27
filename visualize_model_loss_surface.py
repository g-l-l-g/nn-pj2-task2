# VGG_BatchNorm/visualize_model_loss_surface.py
import os
from pathlib import Path

import torch
import torch.nn as nn
import json
from datetime import datetime

# 项目特定导入
from VGG_BatchNorm.data_loader.loaders import get_cifar_loader
from VGG_BatchNorm.models.vgg import VGG_A, VGG_A_BatchNorm
# 假设其他VGG变体 (VGG_A_Light, VGG_A_Dropout) 也在 vgg.py 中定义
from VGG_BatchNorm.utils.loss_surface_3d import plot_3d_loss_surface_with_grad

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_ROOT_DATA_DIR = './data_cifar'  # CIFAR-10 数据集默认路径
DEFAULT_OUTPUT_DIR_BASE = f'./visualizations/3d_loss_surfaces/{datetime.now().strftime("%Y%m%d-%H%M%S")}'


def load_model_and_config(config_path, weights_path, device):
    """
    加载模型配置和权重。
    关键：此函数现在从用户直接指定的 config_path 和 weights_path 读取。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"权重文件未找到: {weights_path}")

    model_name_in_experiment = os.path.basename(config_path)
    if model_name_in_experiment.endswith("_config.json"):
        model_name_in_experiment = model_name_in_experiment[:-len("_config.json")]
    else: # 如果不遵循模式，则使用不带 .json 扩展名的文件名
        model_name_in_experiment = os.path.splitext(model_name_in_experiment)[0]
    with open(config_path, 'r') as f:
        saved_config = json.load(f)

    print(f"已加载模型 '{model_name_in_experiment}' 的配置: {saved_config} from {config_path}")

    # --- 模型实例化逻辑 ---
    num_classes = 10  # 对于 CIFAR-10
    saved_model_params_for_init = saved_config.get("model_params", {})
    init_weights_flag = saved_model_params_for_init.get("init_weights_flag", True)  # 提供一个默认值

    model_instantiation_params = {
        "num_classes": num_classes,
        "init_weights_flag": init_weights_flag
    }

    saved_model_class_name = saved_config.get("model_class")
    if not saved_model_class_name:
        raise ValueError(f"配置文件 '{config_path}' 中缺少 'model_class' 字段。")

    model_class = None
    model_class_name_str = saved_model_class_name

    if saved_model_class_name == "VGG_A_BatchNorm":
        model_class = VGG_A_BatchNorm
        saved_model_params = saved_config.get("model_params")
        if not saved_model_params:
            raise ValueError(f"配置文件中 VGG_A_BatchNorm 的 'model_params' 字段缺失。")

        batch_norm_2d = saved_model_params.get("batch_norm_2d")
        batch_norm_1d = saved_model_params.get("batch_norm_1d")

        if batch_norm_2d is None or batch_norm_1d is None:
            raise ValueError("VGG_A_BatchNorm 的 'model_params' 中缺少 'batch_norm_2d' 或 'batch_norm_1d'。")

        model_instantiation_params["batch_norm_2d"] = batch_norm_2d
        model_instantiation_params["batch_norm_1d"] = batch_norm_1d

    elif saved_model_class_name == "VGG_A":
        model_class = VGG_A
    # 其他模型的分支由于实验并不需要使用而暂未具体实现
    # elif saved_model_class_name == "VGG_A_Light":
    #     from VGG_BatchNorm.models.vgg import VGG_A_Light
    #     model_class = VGG_A_Light
    # elif saved_model_class_name == "VGG_A_Dropout":
    #     from VGG_BatchNorm.models.vgg import VGG_A_Dropout
    #     model_class = VGG_A_Dropout
    else:
        try:
            module = __import__("VGG_BatchNorm.models.vgg", fromlist=[saved_model_class_name])
            model_class = getattr(module, saved_model_class_name)
            print(f"动态加载模型类: {saved_model_class_name}")
        except (ImportError, AttributeError) as e:
            raise ValueError(f"无法加载或找到模型类 '{saved_model_class_name}'。"
                             f"请确保它在 'VGG_BatchNorm.models.vgg' 中定义，或者在此处添加特定的实例化逻辑。错误: {e}")

    if model_class is None:
        raise ValueError(f"未能根据配置中的类名 '{saved_model_class_name}' 确定模型类。")

    print(f"正在实例化模型 {model_class_name_str} 使用参数: {model_instantiation_params}")
    model = model_class(**model_instantiation_params).to(device)

    model.init_params_for_surface_plot = model_instantiation_params.copy()

    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"已从 {weights_path} 加载权重到模型 {model_class_name_str}")
    return model, saved_config, model_name_in_experiment


def main_logic(
        config_file_path,  # 新增
        weights_file_path,  # 新增
        output_dir,
        data_dir,
        n_points,
        range_scale_alpha,
        range_scale_beta,
        num_batches_surface,
        val_items,
        epoch
):
    """
    包含原 main 函数核心逻辑的函数，接收一个包含所有参数的对象。
    """
    print(f"使用设备: {DEVICE}")

    # 从配置文件名推断基础模型名，用于输出目录
    base_model_name_for_output = os.path.basename(config_file_path)
    if base_model_name_for_output.endswith("_config.json"):
        base_model_name_for_output = base_model_name_for_output[:-len("_config.json")]
    else:
        base_model_name_for_output = os.path.splitext(base_model_name_for_output)[0]


    # 确定输出目录
    if output_dir is None:
        plot_output_dir = os.path.join(DEFAULT_OUTPUT_DIR_BASE, base_model_name_for_output)
    else:
        # 在指定的 output_dir下创建以模型名命名的子文件夹
        plot_output_dir = os.path.join(output_dir, base_model_name_for_output)
    os.makedirs(plot_output_dir, exist_ok=True)
    print(f"绘图结果将保存到: {plot_output_dir}")

    # 1. 加载模型及其原始配置
    try:
        model, saved_exp_config, loaded_model_name = load_model_and_config(
            config_file_path, weights_file_path, DEVICE
        )
        model.eval()
    except Exception as e:
        print(f"加载模型或配置时出错: {e}")
        return

    # 2. 加载数据
    print(f"\n正在加载 CIFAR-10 验证数据 (使用 {val_items} 个样本)...")
    hparams_from_config = saved_exp_config.get("input_hyperparameters", {})
    if not isinstance(hparams_from_config, dict):
        print("警告：配置中的 'input_hyperparameters' 不是预期的字典格式，将使用默认 batch_size。")
        hparams_from_config = {}
    batch_size_for_surface = hparams_from_config.get('batch_size', 128)

    surface_dataloader = get_cifar_loader(
        root=data_dir,
        batch_size=batch_size_for_surface,
        train=False,
        shuffle=True,
        num_workers=2,
        n_items=val_items
    )
    if len(surface_dataloader.dataset) == 0:
        print("错误: 用于损失表面的数据加载器为空。请检查数据路径或 val_items 设置。")
        return
    print(f"用于损失表面的数据加载器: {len(surface_dataloader.dataset)} 个样本, {len(surface_dataloader)} 个批次。")

    # 3. 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 4. 绘制 3D 损失表面
    plot_3d_loss_surface_with_grad(
        model_instance_at_center=model,
        dataloader=surface_dataloader,
        criterion=criterion,
        device=DEVICE,
        output_dir=plot_output_dir,
        exp_name=loaded_model_name,
        n_points=n_points,
        range_scale_alpha=range_scale_alpha,
        range_scale_beta=range_scale_beta,
        num_batches_for_loss_surface=num_batches_surface,
        epoch_tag=f"_epoch_{epoch}"
    )

    print("\n3D 损失表面可视化脚本执行完毕。")
    print(f"请在以下目录查看绘图结果: {plot_output_dir}")


if __name__ == '__main__':

    ROOT_DIR = "./runs/bn_comparison_experiments/20250527-125325/saved_models"
    OUTPUT_ROOT_DIR = "./visualizations/3d_loss_surface"
    EXPERIMENT_NAMES = ["VGG_A_lr_1e-3", "VGG_A_lr_5e-4", "VGG_A_BN_2d_only_lr_1e-3", "VGG_A_BN_2d_only_lr_5e-4"]

    num_epochs = 20
    n_points_val = 2
    range_scale_alpha_val = 0.01
    range_scale_beta_val = 0.01
    num_batches_surface_val = 1
    val_items_val = 10
    stride = 6

    data_dir_val = DEFAULT_ROOT_DATA_DIR
    for exp_name in EXPERIMENT_NAMES:
        files_path = os.path.join(ROOT_DIR, exp_name)
        output_dir_val = os.path.join(OUTPUT_ROOT_DIR, exp_name)
        for i in range(stride, num_epochs, stride):
            weights_file_path_val = os.path.join(files_path, f"epoch_{i}.pth")
            config_file_path_val = os.path.join(files_path, f"{exp_name}_config.json")
            main_logic(
                config_file_path=config_file_path_val,
                weights_file_path=weights_file_path_val,
                output_dir=output_dir_val,
                data_dir=data_dir_val,
                n_points=n_points_val,
                range_scale_alpha=range_scale_alpha_val,
                range_scale_beta=range_scale_beta_val,
                num_batches_surface=num_batches_surface_val,
                val_items=val_items_val,
                epoch=i,
            )
