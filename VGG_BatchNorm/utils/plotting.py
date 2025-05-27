# utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_summary(epoch_losses_vgga, train_accuracies_vgga, val_accuracies_vgga,
                          epoch_losses_vgg_bn, train_accuracies_vgg_bn, val_accuracies_vgg_bn,
                          epochs_range, save_path_base, model_name_vgga="VGG-A", model_name_vgg_bn="VGG-A_BN"):
    """
    Plots and saves the training summary (loss and accuracy) for two models.
    """
    plt.figure(figsize=(12, 10))

    # Plot loss
    plt.subplot(2, 1, 1)
    if epoch_losses_vgga:
        plt.plot(epochs_range, epoch_losses_vgga, 'b-', label=f'{model_name_vgga} Average Epoch Loss')
    if epoch_losses_vgg_bn:
        plt.plot(epochs_range, epoch_losses_vgg_bn, 'r-', label=f'{model_name_vgg_bn} Average Epoch Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Average Epoch Loss Comparison')
    plt.legend()
    plt.grid(True)

    # Plot accuracy
    plt.subplot(2, 1, 2)
    if val_accuracies_vgga:
        plt.plot(epochs_range, val_accuracies_vgga, 'b--', label=f'{model_name_vgga} Validation Accuracy')
    if val_accuracies_vgg_bn:
        plt.plot(epochs_range, val_accuracies_vgg_bn, 'r--', label=f'{model_name_vgg_bn} Validation Accuracy')
    if train_accuracies_vgga:
        plt.plot(epochs_range, train_accuracies_vgga, 'b:', label=f'{model_name_vgga} Training Accuracy')
    if train_accuracies_vgg_bn:
        plt.plot(epochs_range, train_accuracies_vgg_bn, 'r:', label=f'{model_name_vgg_bn} Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    comparison_plot_path = os.path.join(save_path_base, "training_summary_comparison.png")
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"Training summary comparison plot saved to {comparison_plot_path}")


def plot_loss_landscape(loss_data_dict_vgga, loss_data_dict_vggbn,
                        model_name_vgga, model_name_vggbn,
                        save_path_base):
    """
    绘制两个模型在不同学习率下的损失状况（最小/最大损失曲线）。
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        print("Seaborn 样式 'seaborn-v0_8-darkgrid' 未找到，使用默认样式。")

    plt.figure(figsize=(10, 6))

    plot_vgga = bool(loss_data_dict_vgga and any(loss_data_dict_vgga.values()))
    plot_vggbn = bool(loss_data_dict_vggbn and any(loss_data_dict_vggbn.values()))

    max_y_limit_for_plot = 0
    min_x_limit_for_plot = float('inf')

    # 绘制 VGG-A (model_name_vgga) 的损失范围
    if plot_vgga:
        valid_losses_vgga = [l for l in loss_data_dict_vgga.values() if l and len(l) > 0]
        if valid_losses_vgga:
            # 确保所有损失列表长度一致，取最短的
            min_len_vgga = min(len(l) for l in valid_losses_vgga)
            all_series_vgga = np.array([l[:min_len_vgga] for l in valid_losses_vgga])

            if all_series_vgga.size > 0:
                max_curve_vgga = np.max(all_series_vgga, axis=0)
                min_curve_vgga = np.min(all_series_vgga, axis=0)
                steps_vgga = np.arange(min_len_vgga)

                # 填充区域，并使用 model_name_vgga 作为标签
                plt.fill_between(steps_vgga, min_curve_vgga, max_curve_vgga,
                                 color='mediumseagreen', alpha=0.5,  # 匹配示例图的绿色
                                 label=model_name_vgga)

                if max_curve_vgga.size > 0: max_y_limit_for_plot = max(max_y_limit_for_plot, np.max(max_curve_vgga))
                min_x_limit_for_plot = min(min_x_limit_for_plot, min_len_vgga)
        else:
            print(f"没有用于绘制损失状况的有效损失数据: {model_name_vgga}")

    # 绘制 VGG-A_BN (model_name_vggbn) 的损失范围
    if plot_vggbn:
        valid_losses_vggbn = [l for l in loss_data_dict_vggbn.values() if l and len(l) > 0]
        if valid_losses_vggbn:
            min_len_vggbn = min(len(l) for l in valid_losses_vggbn)
            all_series_vggbn = np.array([l[:min_len_vggbn] for l in valid_losses_vggbn])

            if all_series_vggbn.size > 0:
                max_curve_vggbn = np.max(all_series_vggbn, axis=0)
                min_curve_vggbn = np.min(all_series_vggbn, axis=0)
                steps_vggbn = np.arange(min_len_vggbn)

                plt.fill_between(steps_vggbn, min_curve_vggbn, max_curve_vggbn,
                                 color='lightcoral', alpha=0.6,  # 匹配示例图的红色
                                 label=model_name_vggbn)

                if max_curve_vggbn.size > 0: max_y_limit_for_plot = max(max_y_limit_for_plot, np.max(max_curve_vggbn))
                min_x_limit_for_plot = min(min_x_limit_for_plot, min_len_vggbn)
        else:
            print(f"没有用于绘制损失状况的有效损失数据: {model_name_vggbn}")

    if not plot_vgga and not plot_vggbn:
        print("没有可用于绘制损失状况图的数据。")
        plt.close()
        return

    plt.xlabel('Steps')  # X轴标签
    plt.ylabel('Loss Landscape')  # Y轴标签，根据示例图修改
    plt.title('Loss Landscape')  # 图标题，根据示例图修改
    plt.legend(loc='upper right')  # 图例位置
    # plt.grid(True, linestyle=':', alpha=0.7) # Seaborn 样式通常自带网格

    # 动态调整Y轴上限，使其更美观
    if max_y_limit_for_plot > 0:
        # 如果最大损失值小于3，给多一点空间；否则，稍微增加一点空间
        plt.ylim(0, max_y_limit_for_plot * 1.1 if max_y_limit_for_plot < 2.5 else max_y_limit_for_plot * 1.05)
    else:
        plt.ylim(0, 2.5)  # 默认Y轴范围，参考示例图

    # 确保X轴有意义
    if min_x_limit_for_plot != float('inf') and min_x_limit_for_plot > 0:
        plt.xlim(0, min_x_limit_for_plot)
    else:
        plt.xlim(0, 100)  # 如果没有数据，默认X轴范围

    figure_save_path = os.path.join(save_path_base, "loss_landscape_comparison.png")
    plt.savefig(figure_save_path)
    plt.close()
    print(f"损失状况比较图已保存到: {figure_save_path}")


def plot_interim_training_progress(epoch_losses, train_accuracies, val_accuracies,
                                   current_epoch, total_epochs, model_name, save_path_base):
    """
    Plots and saves the interim training progress for a single model.
    Called within the training loop.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epoch_losses, color=color, label='Average Epoch Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.7)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(train_accuracies, color='blue', linestyle='--', label='Training Accuracy')
    ax2.plot(val_accuracies, color='green', linestyle='-', label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # Adjust layout to make room for the title and labels
    plt.title(f'Training Progress: {model_name} (Epoch {current_epoch}/{total_epochs})')

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plot_save_path = os.path.join(save_path_base, f'{model_name}_training_progress_epoch_{current_epoch}.png')
    plt.savefig(plot_save_path)
    plt.close(fig)
    # print(f"Interim training progress plot saved to {plot_save_path}") # Optional: can be too verbose


def plot_single_experiment_results(model_name: str, results: dict, base_save_path: str, num_epochs: int):
    """
    Plots and saves loss and accuracy curves for a single experiment in its own directory.
    """
    experiment_plot_dir = os.path.join(base_save_path, model_name)
    os.makedirs(experiment_plot_dir, exist_ok=True)

    epochs_range = range(1, num_epochs + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, results["train_losses"], label='Train Loss', color='blue', linestyle=':')
    plt.plot(epochs_range, results["val_losses"], label='Validation Loss', color='blue', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves - {model_name}')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(experiment_plot_dir, f"{model_name}_loss_curves.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved loss plot for {model_name} to {loss_plot_path}")

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, results["train_accuracies"], label='Train Accuracy', color='red', linestyle=':')
    plt.plot(epochs_range, results["val_accuracies"], label='Validation Accuracy', color='red', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy Curves - {model_name}')
    plt.legend()
    plt.grid(True)
    acc_plot_path = os.path.join(experiment_plot_dir, f"{model_name}_accuracy_curves.png")
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Saved accuracy plot for {model_name} to {acc_plot_path}")
