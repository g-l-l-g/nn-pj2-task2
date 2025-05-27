# VGG_BatchNorm/utils/loss_surface_3d.py
import os
import torch
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _get_trainable_float_param_tensors(model_instance):
    params = []
    names = []
    for name, param in model_instance.named_parameters():
        if param.requires_grad and param.is_floating_point():
            params.append(param.data.clone())  # Store a clone of the parameter data
            names.append(name)
    return params, names


# Helper function: Generate random direction vectors
def _generate_random_direction(param_tensors_list, device):
    """
    Generates random direction vectors, normalized by the norm of each parameter tensor.
    Input: param_tensors_list - A list of tensors (trainable, float parameters from the model).
    """
    direction = []
    for p_tensor in param_tensors_list:
        random_v = torch.randn_like(p_tensor, device=device)
        norm_p_tensor = torch.norm(p_tensor.float())
        norm_random_v = torch.norm(random_v.float())

        if norm_random_v > 1e-10 and norm_p_tensor > 1e-10:  # Avoid division by zero
            random_v_normalized = random_v * (norm_p_tensor / norm_random_v)
        else:
            # If norms are too small, do not scale or use zero (depending on desired behavior)
            random_v_normalized = random_v
        direction.append(random_v_normalized)
    return direction


# Helper function: Calculate loss and gradient norm at a point in parameter space
def _calculate_loss_and_grad_norm_at_point(model, dataloader, criterion, device, num_batches_for_loss_surface=5):
    model.eval()  # Crucial: ensure BN/Dropout layers are in evaluation mode

    # Save original requires_grad state, as we need to compute gradients
    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(True)  # Ensure all parameters compute gradients

    model.zero_grad()  # Clear old gradients

    running_loss = 0.0
    total_samples = 0
    accumulated_grads_sum = [torch.zeros_like(p.data) for p in model.parameters() if p.requires_grad]

    if len(dataloader.dataset) == 0:
        print("Warning: Dataloader for loss surface is empty.")
        return float('nan'), float('nan')

    num_batches_to_eval = min(num_batches_for_loss_surface, len(dataloader))
    if num_batches_to_eval == 0 and len(dataloader) > 0 and num_batches_for_loss_surface > 0:
        num_batches_to_eval = 1  # Evaluate at least one batch if possible
    elif num_batches_to_eval == 0:
        print("Warning: num_batches_for_loss_surface is 0 or dataloader is empty.")
        return float('nan'), float('nan')

    actual_batches_processed = 0
    with torch.enable_grad():  # Ensure gradients are computed in this block
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= num_batches_to_eval:
                break
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss_val = criterion(outputs, labels)

            running_loss += loss_val.item() * inputs.size(0)
            total_samples += inputs.size(0)

            loss_val.backward()  # Backpropagate to compute gradients

            grad_idx = 0
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    accumulated_grads_sum[grad_idx] += param.grad.clone().detach()
                    grad_idx += 1
            model.zero_grad()  # Zero gradients for the next batch
            actual_batches_processed += 1

    if total_samples == 0:
        avg_loss = float('nan')
    else:
        avg_loss = running_loss / total_samples

    final_grad_norm = float('nan')
    if actual_batches_processed > 0 and accumulated_grads_sum:
        # Calculate average gradients
        avg_grads_tensors = [s / actual_batches_processed for s in accumulated_grads_sum]
        # Calculate L2 norm of average gradients
        flat_avg_grads = torch.cat([g.flatten() for g in avg_grads_tensors if g is not None])
        if flat_avg_grads.numel() > 0:  # Ensure there are elements
            final_grad_norm = torch.norm(flat_avg_grads, p=2).item()

    # Restore original requires_grad state
    for name, p in model.named_parameters():
        p.requires_grad_(original_requires_grad[name])

    return avg_loss, final_grad_norm


# Main plotting function
def plot_3d_loss_surface_with_grad(
        model_instance_at_center,  # Actual model object with center weights loaded
        dataloader,
        criterion,
        device,
        output_dir,
        exp_name,
        n_points=10,  # Grid size (NxN)
        range_scale_alpha=0.1,  # Scaling range for direction 1
        range_scale_beta=0.1,  # Scaling range for direction 2
        num_batches_for_loss_surface=5,
        epoch_tag=""  # e.g., "_epoch_10" or "_best"
):
    print(f"\n--- Generating 3D Loss Surface Plot for {exp_name}{epoch_tag} (Grid: {n_points}x{n_points}) ---")
    if dataloader is None or len(dataloader.dataset) == 0:
        print("Dataloader for loss surface is empty, skipping.")
        return
    if n_points < 2:
        print("n_points must be at least 2, skipping.")
        return

    # Get the state of trainable float parameters of the center model
    center_trainable_param_tensors, trainable_param_names = _get_trainable_float_param_tensors(model_instance_at_center)

    if not center_trainable_param_tensors:
        print("Error: Model instance has no trainable float parameters.")
        return

    print("Generating random direction 1...")
    direction1_trainable = _generate_random_direction(center_trainable_param_tensors, device)
    print("Generating random direction 2...")
    direction2_trainable = _generate_random_direction(center_trainable_param_tensors, device)

    # Orthogonalize direction2 with respect to direction1 (Gram-Schmidt)
    dot_product_d1_d2 = sum(torch.sum(d1_p * d2_p) for d1_p, d2_p in zip(direction1_trainable, direction2_trainable))
    norm_sq_d1 = sum(torch.sum(d1_p * d1_p) for d1_p in direction1_trainable)

    if norm_sq_d1 > 1e-10:  # Avoid division by zero
        projection_factor = dot_product_d1_d2 / norm_sq_d1
        for idx in range(len(direction2_trainable)):
            direction2_trainable[idx] -= projection_factor * direction1_trainable[idx]

    alpha_coords = np.linspace(-range_scale_alpha, range_scale_alpha, n_points)
    beta_coords = np.linspace(-range_scale_beta, range_scale_beta, n_points)
    Alpha_grid, Beta_grid = np.meshgrid(alpha_coords, beta_coords)

    losses_surface = np.full_like(Alpha_grid, float('nan'), dtype=float)
    grad_norms_surface = np.full_like(Alpha_grid, float('nan'), dtype=float)

    if not hasattr(model_instance_at_center, 'init_params_for_surface_plot'):
        raise AttributeError("model_instance_at_center is missing 'init_params_for_surface_plot' attribute, "
                             "cannot create temporary model for calculations.")
    temp_model_for_calc = type(model_instance_at_center)(**model_instance_at_center.init_params_for_surface_plot)
    temp_model_for_calc.to(device)

    center_full_state_dict = model_instance_at_center.state_dict()

    print(f"Calculating loss surface on a {n_points}x{n_points} grid...")
    with tqdm(total=n_points * n_points, desc="Loss Surface Calculation") as pbar:
        for i_idx in range(n_points):
            for j_idx in range(n_points):
                alpha = Alpha_grid[j_idx, i_idx]
                beta = Beta_grid[j_idx, i_idx]

                perturbed_full_state_dict = {k: v.clone() for k, v in center_full_state_dict.items()}

                current_param_idx = 0
                for name, _ in temp_model_for_calc.named_parameters():
                    if name in trainable_param_names:
                        original_param_tensor = center_trainable_param_tensors[current_param_idx]
                        d1_tensor = direction1_trainable[current_param_idx]
                        d2_tensor = direction2_trainable[current_param_idx]

                        perturbed_tensor = original_param_tensor + alpha * d1_tensor + beta * d2_tensor
                        perturbed_full_state_dict[name] = perturbed_tensor
                        current_param_idx += 1

                try:
                    temp_model_for_calc.load_state_dict(perturbed_full_state_dict)
                    loss, grad_norm = _calculate_loss_and_grad_norm_at_point(
                        temp_model_for_calc, dataloader, criterion, device, num_batches_for_loss_surface
                    )
                    losses_surface[j_idx, i_idx] = loss
                    grad_norms_surface[j_idx, i_idx] = grad_norm
                except Exception as e:
                    print(f"Error at alpha={alpha}, beta={beta}: {e}")
                    losses_surface[j_idx, i_idx] = float('nan')
                    grad_norms_surface[j_idx, i_idx] = float('nan')

                pbar.set_postfix({
                    "loss": f"{losses_surface[j_idx, i_idx]:.4f}",
                    "gradN": f"{grad_norms_surface[j_idx, i_idx]:.4f}"
                })
                pbar.update(1)

    # --- Plotly Interactive Plot ---
    print("Plotting interactive 3D loss surface with Plotly...")
    center_loss_val, center_grad_norm_val = _calculate_loss_and_grad_norm_at_point(
        model_instance_at_center, dataloader, criterion, device, num_batches_for_loss_surface
    )

    # Hover template
    hover_text_template = (
        '<b>Loss</b>: %{z:.4f}<br>'
        '<b>Gradient Norm (L2)</b>: %{customdata[0]:.4e}<br>'  # customdata[0] corresponds to grad_norms_surface
        'Alpha: %{x:.3f}<br>'
        'Beta: %{y:.3f}'
        '<extra></extra>'  # Hide default trace name
    )

    # Stack grad_norms_surface to pass as customdata (needs same shape as Z)
    custom_data_for_hover = np.stack([grad_norms_surface], axis=-1)  # shape (n_points, n_points, 1)

    fig_plotly = go.Figure(data=[go.Surface(
        z=losses_surface,
        x=Alpha_grid,
        y=Beta_grid,
        colorscale='Viridis',
        customdata=custom_data_for_hover,  # Pass gradient norm data
        hovertemplate=hover_text_template,  # Use custom hover template
        colorbar=dict(title='Loss Value'),
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)),
        name='Loss Surface'
    )])

    # Add center point
    if not np.isnan(center_loss_val):
        fig_plotly.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[center_loss_val], mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            customdata=np.array([[center_grad_norm_val]]),  # customdata needs to be 2D
            hovertemplate=(
                '<b>Center Point</b><br>'
                'Loss: %{z:.4f}<br>'
                'Gradient Norm (L2): %{customdata[0]:.4e}<br>'
                'Alpha: 0, Beta: 0'
                '<extra></extra>'
            ),
            name=f'Center (Loss: {center_loss_val:.4f}, GradN: {center_grad_norm_val:.2e})'
        ))

    # Add minimum loss point on the surface
    min_loss_val_surface = np.nanmin(losses_surface)
    min_alpha_surface, min_beta_surface, min_grad_norm_at_min_loss = None, None, None
    if not np.isnan(min_loss_val_surface):
        min_idx = np.unravel_index(np.nanargmin(losses_surface), losses_surface.shape)
        min_alpha_surface, min_beta_surface = Alpha_grid[min_idx], Beta_grid[min_idx]
        min_grad_norm_at_min_loss = grad_norms_surface[min_idx]
        fig_plotly.add_trace(go.Scatter3d(
            x=[min_alpha_surface], y=[min_beta_surface], z=[min_loss_val_surface], mode='markers',
            marker=dict(size=8, color='cyan', symbol='circle'),
            customdata=np.array([[min_grad_norm_at_min_loss]]),
            hovertemplate=(
                '<b>Surface Minimum Loss</b><br>'
                'Loss: %{z:.4f}<br>'
                'Gradient Norm (L2): %{customdata[0]:.4e}<br>'
                'Alpha: %{x:.3f}, Beta: %{y:.3f}'
                '<extra></extra>'
            ),
            name=f'Surface Min (Loss: {min_loss_val_surface:.4f}, GradN: {min_grad_norm_at_min_loss:.2e})'
        ))

    fig_plotly.update_layout(
        title=f'Interactive 3D Loss Surface: {exp_name}{epoch_tag}<br>(Grid: {n_points}x{n_points}, '
              f'α Range: {range_scale_alpha}, β Range: {range_scale_beta})',
        scene=dict(
            xaxis_title='Alpha (Direction 1)',
            yaxis_title='Beta (Direction 2)',
            zaxis_title='Loss Value',
            camera_eye=dict(x=1.7, y=1.7, z=1.2)  # Adjust camera view
        ),
        autosize=True, margin=dict(l=50, r=50, b=50, t=100),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)  # Legend position
    )

    os.makedirs(output_dir, exist_ok=True)
    plot_filename_html = (f"loss_surface_3d_interactive_{exp_name.replace(' ', '_')}"
                          f"{epoch_tag}_{n_points}x{n_points}.html")
    plot_path_html = os.path.join(output_dir, plot_filename_html)
    try:
        fig_plotly.write_html(plot_path_html, include_plotlyjs='cdn')  # 'cdn' makes html file smaller
        print(f"Interactive 3D loss surface plot saved to: {plot_path_html}")
    except Exception as e:
        print(f"Error saving interactive 3D loss surface plot: {e}")

    # --- Matplotlib Static Plot ---
    fig_static = None
    try:
        print("Generating Matplotlib static 3D loss surface plot...")
        fig_static = plt.figure(figsize=(12, 9))  # Adjust figure size
        ax_static = fig_static.add_subplot(111, projection='3d')

        Z_masked = np.ma.masked_invalid(losses_surface)  # Mask NaN values for plotting
        surf_static = ax_static.plot_surface(Alpha_grid, Beta_grid, Z_masked, cmap='viridis', edgecolor='none',
                                             rstride=1, cstride=1, alpha=0.9)

        ax_static.set_xlabel(f'Alpha (Range: {range_scale_alpha})')
        ax_static.set_ylabel(f'Beta (Range: {range_scale_beta})')
        ax_static.set_zlabel('Loss Value')

        # Calculate statistics for gradient norms
        valid_grad_norms = grad_norms_surface[~np.isnan(grad_norms_surface)]  # Exclude NaN
        avg_grad_norm_surface = np.mean(valid_grad_norms) if len(valid_grad_norms) > 0 else float('nan')
        min_grad_norm_surface = np.min(valid_grad_norms) if len(valid_grad_norms) > 0 else float('nan')
        max_grad_norm_surface = np.max(valid_grad_norms) if len(valid_grad_norms) > 0 else float('nan')

        title_str = (f'Static 3D Loss Surface: {exp_name}{epoch_tag}\n'
                     f'Center - Loss: {center_loss_val:.3f}, Grad Norm: {center_grad_norm_val:.2e}\n'
                     f'Surface Grad Norm - Min: {min_grad_norm_surface:.2e}, '
                     f'Max: {max_grad_norm_surface:.2e}, Avg: {avg_grad_norm_surface:.2e}')
        ax_static.set_title(title_str, fontsize=10)  # Adjust title font size

        if not np.all(np.isnan(losses_surface)):  # Add colorbar only if there's data
            fig_static.colorbar(surf_static, shrink=0.5, aspect=10, label='Loss Value')

        # Add legend elements
        legend_handles = []
        if not np.isnan(center_loss_val):
            p_center = ax_static.scatter([0], [0], [center_loss_val], color='red', s=60, edgecolor='black',
                                         depthshade=False, label=f'Center Point')
            legend_handles.append(p_center)
        if min_alpha_surface is not None:  # If surface minimum was found
            p_min_surf = ax_static.scatter([min_alpha_surface], [min_beta_surface], [min_loss_val_surface],
                                           color='cyan', s=60, edgecolor='black', depthshade=False,
                                           label=f'Surface Min Loss')
            legend_handles.append(p_min_surf)

        if legend_handles:
            ax_static.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.0, 0.9))

        ax_static.view_init(elev=20, azim=-65)  # Adjust view angle

        plot_filename_static = (f"loss_surface_3d_static_{exp_name.replace(' ', '_')}"
                                f"{epoch_tag}_{n_points}x{n_points}.png")
        plot_path_static = os.path.join(output_dir, plot_filename_static)
        fig_static.savefig(plot_path_static, dpi=150, bbox_inches='tight')  # bbox_inches avoids label cutoff
        print(f"Static 3D loss surface plot saved to: {plot_path_static}")
    except Exception as e:
        print(f"Error saving static 3D loss surface plot: {e}")
    finally:
        if fig_static:  # Ensure figure is closed even if an error occurs
            plt.close(fig_static)
