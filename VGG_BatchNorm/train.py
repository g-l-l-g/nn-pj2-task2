# train.py
import os
import torch
import json
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def _evaluate_model_on_epoch(model, data_loader, criterion, device):
    """
    Evaluates the model on the given data loader for one epoch.
    Internal helper function.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    if len(data_loader.dataset) == 0:
        print(f"Warning: Validation dataset is empty. Skipping evaluation.")
        return 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    if total_samples == 0:
        print("Warning: No samples processed during evaluation (total_samples is 0).")
        return 0.0, 0.0

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct_predictions / total_samples
    return avg_loss, accuracy


def train_single_model_experiment(
        model_name: str,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        device: torch.device,
        base_log_dir: str,
        seed: int,
        hparams_dict: dict = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        model_save_dir: str = None
):
    writer_log_dir = os.path.join(base_log_dir, model_name)
    writer = SummaryWriter(log_dir=writer_log_dir)
    print(f"TensorBoard logs for {model_name} will be saved to: {writer_log_dir}")

    if hparams_dict:
        print(f"Starting training for {model_name} with Hyperparameters: {hparams_dict}")

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accuracies = []
    epoch_val_accuracies = []

    best_val_accuracy = 0.0
    # best_model_path = None # Not strictly needed if filename is fixed

    # --- 保存模型配置信息 ---
    if hparams_dict and model_save_dir:
        model_config_to_save = {
            "seed": seed,
            "model_name": model_name,
            "model_class": model.__class__.__name__,
            "criterion": criterion.__class__.__name__,
            "total_epochs_planned_for_run": num_epochs,
            "input_hyperparameters": hparams_dict if hparams_dict else {},
        }
        if model_config_to_save["model_class"] == "VGG_A_BatchNorm":
            model_config_to_save["model_params"] = {
                "batch_norm_1d": model.batch_norm_1d,
                "batch_norm_2d": model.batch_norm_2d,
            }
        config_save_path = os.path.join(model_save_dir, f"{model_name}_config.json")
        try:
            with open(config_save_path, 'w') as f:
                json.dump(model_config_to_save, f, indent=4)
            print(f"Saved model configuration to {config_save_path}")
        except Exception as e:
            print(f"Warning: Could not save model configuration to {config_save_path}. Error: {e}")

    for epoch in tqdm(range(num_epochs), unit='epoch', desc=f"Training {model_name}"):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        if len(train_loader.dataset) == 0:
            print(f"Warning: Training dataset for {model_name} is empty. Skipping training for epoch {epoch + 1}.")
            avg_epoch_train_loss = 0.0
            epoch_train_accuracy = 0.0
        else:
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            avg_epoch_train_loss = running_loss / total_train if total_train > 0 else 0.0
            epoch_train_accuracy = 100.0 * correct_train / total_train if total_train > 0 else 0.0

        epoch_train_losses.append(avg_epoch_train_loss)
        epoch_train_accuracies.append(epoch_train_accuracy)

        writer.add_scalar(f'Loss/train_epoch', avg_epoch_train_loss, epoch)
        writer.add_scalar(f'Accuracy/train_epoch', epoch_train_accuracy, epoch)

        # Evaluation phase
        avg_epoch_val_loss, epoch_val_accuracy = _evaluate_model_on_epoch(model, val_loader, criterion, device)
        epoch_val_losses.append(avg_epoch_val_loss)
        epoch_val_accuracies.append(epoch_val_accuracy)

        writer.add_scalar(f'Loss/validation_epoch', avg_epoch_val_loss, epoch)
        writer.add_scalar(f'Accuracy/validation_epoch', epoch_val_accuracy, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}] for {model_name} | "
              f"Train Loss: {avg_epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}% | "
              f"Val Loss: {avg_epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%")

        # Save best model
        if model_save_dir and epoch_val_accuracy > best_val_accuracy:
            if len(val_loader.dataset) > 0 and total_train > 0:
                best_val_accuracy = epoch_val_accuracy
                current_best_model_path = os.path.join(model_save_dir, "best_model.pth")
                torch.save(model.state_dict(), current_best_model_path)
                print(
                    f"Epoch {epoch + 1}: New best model saved to {current_best_model_path}"
                    f" (Val Acc: {best_val_accuracy:.2f}%)")

                # --- 保存模型配置信息 ---
                model_config_to_save = {
                    "model_name": model_name,
                    "saved_at_epoch": epoch + 1,
                    "total_epochs_planned_for_run": num_epochs,
                    "metrics_at_this_epoch": {
                        "validation_accuracy": epoch_val_accuracy,
                        "validation_loss": avg_epoch_val_loss,
                        "training_accuracy": epoch_train_accuracy,
                        "training_loss": avg_epoch_train_loss,
                    },
                }
                config_save_path = os.path.join(model_save_dir, "best_model_result.json")
                try:
                    with open(config_save_path, 'w') as f:
                        json.dump(model_config_to_save, f, indent=4)
                    print(f"Saved model configuration to {config_save_path}")
                except Exception as e:
                    print(f"Warning: Could not save model configuration to {config_save_path}. Error: {e}")

        # 保存中间过程的模型权重
        current_model_path = os.path.join(model_save_dir, f"epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), current_model_path)

        # Step the scheduler (typically after an epoch)
        if scheduler:
            scheduler.step()
            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LearningRate/epoch', current_lr, epoch)

    if hparams_dict:
        metrics_to_log = {
            "hparam/best_validation_accuracy": best_val_accuracy,
            "hparam/final_train_accuracy": epoch_train_accuracies[-1] if epoch_train_accuracies else 0.0,
            "hparam/final_validation_accuracy": epoch_val_accuracies[-1] if epoch_val_accuracies else 0.0,
            "hparam/final_train_loss": epoch_train_losses[-1] if epoch_train_losses else float('inf'),
            "hparam/final_validation_loss": epoch_val_losses[-1] if epoch_val_losses else float('inf'),
        }

        sanitized_hparams = {}
        for k, v in hparams_dict.items():
            if isinstance(v, (int, float, str, bool, torch.Tensor)):
                sanitized_hparams[k] = v
            elif isinstance(v, type):
                sanitized_hparams[k] = v.__name__
            else:
                try:
                    sanitized_hparams[k] = str(v)
                except Exception:
                    print(f"Warning: Could not convert hparam '{k}' (value: {v})"
                          f" to string for TensorBoard logging. Skipping this hparam.")

        try:
            for k_metric, v_metric in metrics_to_log.items():
                if v_metric is None:
                    metrics_to_log[k_metric] = 0.0 if "accuracy" in k_metric else float('inf')

            writer.add_hparams(sanitized_hparams, metrics_to_log)
        except Exception as e:
            print(f"Warning: Could not log HParams for {model_name} to TensorBoard: {e}")
            print(f"Sanitized HParams: {sanitized_hparams}")
            print(f"Metrics: {metrics_to_log}")

    writer.close()

    return {
        "train_losses": epoch_train_losses,
        "val_losses": epoch_val_losses,
        "train_accuracies": epoch_train_accuracies,
        "val_accuracies": epoch_val_accuracies,
        "best_val_accuracy": best_val_accuracy
    }


def train_and_collect_batch_losses(
        model_run_name: str,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        device: torch.device,
        base_log_dir: str,
):
    writer_log_dir = os.path.join(base_log_dir, "batch_losses_tb", model_run_name)  # 为批次损失创建一个子目录
    writer = SummaryWriter(log_dir=writer_log_dir)
    print(f"TensorBoard 批次日志 ({model_run_name}) 将保存到: {writer_log_dir}")

    all_batch_losses = []
    global_step = 0

    for epoch in tqdm(range(num_epochs), unit='epoch', desc=f"Training {model_run_name}"):
        model.train()
        epoch_batch_losses_for_avg = []  # 用于打印该 epoch 的平均损失

        if len(train_loader.dataset) == 0:
            print(f"警告: {model_run_name} 的训练数据集为空。跳过 epoch {epoch + 1} 的训练。")
            continue

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            all_batch_losses.append(current_loss)
            epoch_batch_losses_for_avg.append(current_loss)
            writer.add_scalar('Loss/train_batch', current_loss, global_step)  # 记录每个批次的损失
            global_step += 1

        avg_epoch_loss = sum(epoch_batch_losses_for_avg) / len(
            epoch_batch_losses_for_avg) if epoch_batch_losses_for_avg else 0
        print(f"Epoch [{epoch + 1}/{num_epochs}] for {model_run_name} | 平均批次损失: {avg_epoch_loss:.4f}")

    writer.close()
    return all_batch_losses
