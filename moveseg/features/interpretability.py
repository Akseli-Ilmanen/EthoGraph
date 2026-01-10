"""
Grad-CAM-like visualization for DiffAct temporal action segmentation model

This script provides gradient-based attention visualization to understand
which parts of the input features and temporal locations the model focuses on.

Integrated with the MovFormer main2.py system for trial-based analysis.
"""

import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from pathlib import Path

from moveseg.diffact.model import ASDiffusionModel
from moveseg.utils.dataset import VideoFeatureDataset, get_trial_dict, get_data_dict
from moveseg.diffact.utils import load_config_file





class DiffActGradCAM:
    def __init__(self, model, device, target_layer_name='encoder.encoder.layers.7'):
        """
        Grad-CAM for DiffAct model
        
        Args:
            model: Trained DiffAct model
            device: torch device
            target_layer_name: Name of layer to hook (default: middle encoder layer)
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer = self._get_layer_by_name(target_layer_name)
        self.forward_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = self.target_layer.register_backward_hook(self._backward_hook)
        
    def _get_layer_by_name(self, layer_name):
        """Get layer by dot notation name"""
        layer = self.model
        for name in layer_name.split('.'):
            layer = getattr(layer, name)
        return layer
    
    def _forward_hook(self, module, input, output):
        """Hook to capture forward activations"""
        self.activations = output.detach()
        
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook to capture backward gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, video_features, target_class=None, target_time=None, mode='encoder'):
        """
        Generate Class Activation Map for given input
        
        Args:
            video_features: Input features [1, F, T]
            target_class: Target class index (if None, uses predicted class)
            target_time: Target time index (if None, uses all timesteps)
            mode: 'encoder' or 'decoder'
        
        Returns:
            cam: Class activation map [T, F]
            prediction: Model prediction
        """
        video_features = video_features.to(self.device)
        video_features.requires_grad_(True)
        
        # Forward pass
        if mode == 'encoder':
            output = self.model.encoder(video_features)
        else:
            output = self.model.ddim_sample(video_features)
        
        # Get target class
        if target_class is None:
            target_class = torch.argmax(output.mean(dim=2), dim=1).item()
        
        # Get target time range
        if target_time is None:
            target_time = slice(None)  # All timesteps
        elif isinstance(target_time, int):
            target_time = slice(target_time, target_time + 1)
        
        # Compute loss for target class and time
        target_output = output[:, target_class, target_time]
        loss = target_output.mean()
        
        # Backward pass
        loss.backward()
        
        # Generate CAM
        if self.gradients is not None and self.activations is not None:
            # Global average pooling of gradients
            weights = torch.mean(self.gradients, dim=2, keepdim=True)  # [1, C, 1]
            
            # Weighted combination of activations
            cam = torch.sum(weights * self.activations, dim=1).squeeze(0)  # [T]
            
            # Apply ReLU and normalize
            cam = F.relu(cam)
            if cam.max() > 0:
                cam = cam / cam.max()
        else:
            # Fallback: use input gradients
            input_gradients = video_features.grad
            cam = torch.mean(torch.abs(input_gradients), dim=1).squeeze(0)  # [T]
            if cam.max() > 0:
                cam = cam / cam.max()
        
        return cam.cpu().numpy(), output.detach().cpu()
    
    def generate_feature_importance(self, video_features, target_class=None):
        """
        Generate feature importance map across all features and time
        
        Returns:
            importance_map: [F, T] importance scores
        """
        video_features = video_features.to(self.device)
        video_features.requires_grad_(True)
        
        # Forward pass
        output = self.model.encoder(video_features)
        
        # Get target class
        if target_class is None:
            target_class = torch.argmax(output.mean(dim=2), dim=1).item()
        
        # Loss for target class
        loss = output[:, target_class, :].mean()
        loss.backward()
        
        # Feature importance = absolute gradient
        importance_map = torch.abs(video_features.grad).squeeze(0)  # [F, T]
        
        # Normalize
        if importance_map.max() > 0:
            importance_map = importance_map / importance_map.max()
        
        return importance_map.cpu().numpy(), target_class
    
    def visualize_temporal_attention(self, video_features, video_name, action_labels=None, 
                                   save_path=None, target_classes=None):
        """
        Create comprehensive visualization of model attention
        
        Args:
            video_features: Input features [1, F, T] 
            video_name: Name of the video
            action_labels: Ground truth labels [T]
            save_path: Path to save visualization
            target_classes: List of classes to analyze (if None, uses top predicted classes)
        """
        # Get model prediction
        with torch.no_grad():
            output = self.model.encoder(video_features.to(self.device))
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze().cpu().numpy()
        
        # Get target classes to analyze (all except class zero)
        if target_classes is None:
            target_classes = np.unique(predictions)
            target_classes = target_classes[target_classes != 0]


        # Generate importance map
        importance_map, _ = self.generate_feature_importance(video_features)

        class_colors = [
            [255, 102, 178],
            [102, 158, 255],
            [153, 51, 255],
            [255, 51, 51],
            [102, 255, 102],
            [255, 153, 102],
            [0, 153, 0],
            [0, 0, 128],
            [255, 255, 0],
            [255, 0, 255],
            [128, 128, 0],
            [0, 204, 204],
            [255, 102, 178],
            [255, 165, 0]
        ]
        class_colors = [[r/255, g/255, b/255] for r, g, b in class_colors]

        # Create visualization
        fig = plt.figure(figsize=(20, 12))

        # 1. Feature importance heatmap (Speed plot)
        gs = fig.add_gridspec(10, 1)  # Use a 10-row grid for more flexible height allocation

        # 1. Feature importance summary (average over time, occupy rows 0-2)
        ax1 = fig.add_subplot(gs[0:2, 0])
        feature_importance_avg = np.mean(importance_map, axis=1)
        top_features = np.argsort(feature_importance_avg)[-20:]  # Top 20 features
        bars = ax1.bar(range(len(top_features)), feature_importance_avg[top_features])
        ax1.set_title('Top 20 Most Important Features (Averaged over Time)', fontsize=14)
        ax1.set_ylabel('Average Importance')
        ax1.set_xlabel('Feature Rank')
        ax1.set_xticks(range(len(top_features)))
        ax1.set_xticklabels([f'F{idx}' for idx in top_features], rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. Speed plot (occupy rows 2-4)
        ax2 = fig.add_subplot(gs[2:4, 0])
        speed = video_features[0, 3, :].detach().cpu().numpy()
        ax2.plot(np.arange(speed.shape[0]), speed, label='Speed', color='blue', linewidth=2, alpha=0.7)
        ax2.set_title(f'Speed - {video_name}', fontsize=14)
        ax2.set_ylabel('Speed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), visible=False)  # Hide x tick labels for alignment

        # 3. Temporal attention for different classes (occupy rows 4-7)
        ax3 = fig.add_subplot(gs[4:7, 0], sharex=ax2)
        for i, target_class in enumerate(target_classes):
            cam, _ = self.generate_cam(video_features, target_class=target_class)
            ax3.plot(np.arange(cam.shape[0]), cam, label=f'Class {target_class}', color=class_colors[i], linewidth=2)
        ax3.set_title(f'Feed forward output after layer 7 for Different Behav Classes', fontsize=14)
        ax3.set_ylabel('Attention Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.get_xticklabels(), visible=False)  # Hide x tick labels for alignment

        # only show 20 first features for visualization
        if importance_map.shape[0] > 20:
            # Take the first 16 features, and then the top 9 from top_features (excluding any overlap)
            first_16 = np.arange(16)
            # Exclude any indices already in first_16 from top_features
            top_9 = [idx for idx in top_features[::-1] if idx not in first_16][:9]
            selected_features = np.concatenate([first_16, top_9])
            importance_map = importance_map[selected_features, :]
        else:
            selected_features = np.arange(importance_map.shape[0])

        # 4. Feature importance heatmap (occupy rows 7-10)
        ax4 = fig.add_subplot(gs[7:10, 0], sharex=ax2)
        im2 = ax4.imshow(importance_map, aspect='auto', cmap='hot', interpolation='nearest', extent=[0, importance_map.shape[1], -0.5, len(selected_features)-0.5])

        row_groups = [(0, 4), (5, 8), (9, 15), (16, 25)]
        for start, end in row_groups:
            if start < len(selected_features):
                rect_end = min(end, len(selected_features)-1)
                rect = Rectangle(
                    (0 - 0.5, start - 0.5),  # (x, y) start at left edge, top of start row
                    importance_map.shape[1],  # width (number of columns)
                    rect_end - start + 1,     # height (number of rows in group)
                    linewidth=2,
                    edgecolor='cyan',
                    facecolor='none'
                )
                ax4.add_patch(rect)

        ax4.set_title(f'Feature Importance Map - {video_name}', fontsize=14)
        ax4.set_ylabel('Feature Index')
        ax4.set_xlabel('Time')
        # Set y-ticks to selected_features values
        ax4.set_yticks(np.arange(len(selected_features)))
        ax4.set_yticklabels([str(idx) for idx in selected_features])

        # Align x-axis ticks across plots 2-4
        for ax in [ax2, ax3, ax4]:
            ax.set_xlim([0, importance_map.shape[1]])

        # Set the same x-ticks for all plots 2-4
        n_time = importance_map.shape[1]
        xticks = np.linspace(0, n_time-1, min(11, n_time), dtype=int)
        ax4.set_xticks(xticks)
        ax4.set_xticklabels([str(x) for x in xticks])
        ax3.set_xticks(xticks)
        ax2.set_xticks(xticks)



        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        # Close the figure to free memory
        plt.close(fig)
        
        return importance_map, predictions
    
    def analyze_action_segments(self, video_features, action_labels, video_name, save_path=None):
        """
        Analyze attention patterns for different action segments
        """
        # Get unique actions and their segments
        unique_actions = np.unique(action_labels)
        
        fig, axes = plt.subplots(len(unique_actions), 2, figsize=(16, 4 * len(unique_actions)))
        if len(unique_actions) == 1:
            axes = axes.reshape(1, -1)
        
        for i, action_class in enumerate(unique_actions):
            # Find segments of this action
            action_mask = (action_labels == action_class)
            action_indices = np.where(action_mask)[0]
            
            if len(action_indices) == 0:
                continue
            
            # Generate CAM for this action
            cam, _ = self.generate_cam(video_features, target_class=action_class)
            
            # Plot 1: CAM over time with action segments highlighted
            ax1 = axes[i, 0]
            ax1.plot(cam, linewidth=2, color='blue', alpha=0.7)
            
            # Highlight action segments
            segments = []
            start = action_indices[0]
            for j in range(1, len(action_indices)):
                if action_indices[j] != action_indices[j-1] + 1:
                    segments.append((start, action_indices[j-1]))
                    start = action_indices[j]
            segments.append((start, action_indices[-1]))
            
            for start, end in segments:
                ax1.axvspan(start, end, alpha=0.3, color='red', label='Action Segment' if start == segments[0][0] else "")
            
            ax1.set_title(f'Action Class {action_class}: Temporal Attention')
            ax1.set_ylabel('Attention Score')
            ax1.set_xlabel('Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Feature importance during action segments
            ax2 = axes[i, 1]
            
            # Get feature importance for action segments only
            segment_features = video_features[:, :, action_indices]
            importance_map, _ = self.generate_feature_importance(segment_features, target_class=action_class)
            
            # Average over time for this action
            avg_importance = np.mean(importance_map, axis=1)
            top_features = np.argsort(avg_importance)[-15:]
            
            bars = ax2.bar(range(len(top_features)), avg_importance[top_features])
            ax2.set_title(f'Action Class {action_class}: Top Features')
            ax2.set_ylabel('Average Importance')
            ax2.set_xlabel('Feature Rank')
            ax2.set_xticks(range(len(top_features)))
            ax2.set_xticklabels([f'F{idx}' for idx in top_features], rotation=45)
            
            # Color by importance
            norm = plt.Normalize(avg_importance[top_features].min(), avg_importance[top_features].max())
            colors = plt.cm.plasma(norm(avg_importance[top_features]))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Action analysis saved to {save_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    def cleanup(self):
        """Remove hooks"""
        self.forward_hook.remove()
        self.backward_hook.remove()


class TrialFeatureImportanceAnalyzer:
    """
    Integrates Grad-CAM analysis with the current MovFormer trial-based system.
    Works with Trainer class and NetCDF datasets.
    """

    def __init__(self, trainer, device, target_layer_name='encoder.encoder.layers.7'):
        """
        Initialize analyzer with existing trained model from Trainer

        Args:
            trainer: Trained Trainer instance from main2.py
            device: torch device
            target_layer_name: Layer to analyze for Grad-CAM
        """
        self.trainer = trainer
        self.device = device
        self.grad_cam = DiffActGradCAM(trainer.model, device, target_layer_name)
        self.event_list = trainer.event_list

    def analyze_trial_importance(self, test_dataset, video_idx, save_dir=None,
                               target_classes=None, mode='decoder-agg'):
        """
        Analyze feature importance for a specific trial

        Args:
            test_dataset: VideoFeatureDataset in test mode
            video_idx: Index of video/trial to analyze
            save_dir: Directory to save results
            target_classes: Classes to analyze (if None, uses predicted classes)
            mode: Model inference mode

        Returns:
            results: Dict with importance maps, predictions, and metadata
        """
        # Get trial data using existing system
        feature, label, boundary, video_name, nc_path, _ = test_dataset[video_idx]

        print(f"Analyzing trial: {video_name}")
        print(f"Feature shapes: {[f.shape for f in feature]}")

        # Use existing test_single_video method to get predictions and confidence
        video_pred, output, gt_label, output_probs, _ = self.trainer.test_single_video(
            video_idx, test_dataset, mode, self.device
        )

        # Convert features to tensor for Grad-CAM (use middle temporal augmentation)
        feature_tensor = feature[len(feature)//2]  # Already [1, F, T] from dataset
        if feature_tensor.dim() == 2:  # If [F, T], add batch dimension
            feature_tensor = feature_tensor.unsqueeze(0)

        # Get feature importance map
        importance_map, predicted_class = self.grad_cam.generate_feature_importance(
            feature_tensor, target_class=None
        )

        # Get temporal attention for different classes
        if target_classes is None:
            unique_preds = np.unique(output)
            target_classes = unique_preds[unique_preds != 0].astype(int).tolist()  # Exclude background, convert to int

        class_attention_maps = {}
        for class_id in target_classes:
            cam, _ = self.grad_cam.generate_cam(feature_tensor, target_class=int(class_id))
            class_attention_maps[class_id] = cam

        # Create comprehensive visualization
        if save_dir:
            save_path = os.path.join(save_dir, f'{video_name}_feature_importance.png')
            self._create_trial_visualization(
                feature_tensor, video_name, importance_map, class_attention_maps,
                output_probs, gt_label, output, save_path
            )

        # Prepare results
        results = {
            'video_name': video_name,
            'nc_path': nc_path,
            'importance_map': importance_map,  # [F, T]
            'class_attention_maps': class_attention_maps,  # {class_id: [T]}
            'predictions': output,  # [T]
            'ground_truth': gt_label,  # [T]
            'output_probabilities': output_probs,  # [C, T]
            'predicted_classes': target_classes,
            'feature_shape': feature_tensor.shape,
            'top_features_per_class': self._get_top_features_per_class(
                importance_map, class_attention_maps
            )
        }

        return results

    def _get_top_features_per_class(self, importance_map, class_attention_maps, top_k=10):
        """Get most important features for each predicted class"""
        top_features = {}

        for class_id, attention in class_attention_maps.items():
            # Weight importance map by class attention
            weighted_importance = importance_map * attention.reshape(1, -1)

            # Average over time to get per-feature importance
            feature_importance = np.mean(weighted_importance, axis=1)

            # Get top features
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            top_features[class_id] = {
                'indices': top_indices,
                'scores': feature_importance[top_indices],
                'class_name': self.event_list[class_id] if class_id < len(self.event_list) else f'Class_{class_id}'
            }

        return top_features

    def _create_trial_visualization(self, feature_tensor, video_name, importance_map,
                                  class_attention_maps, output_probs, gt_label, predictions, save_path):
        """Create comprehensive visualization for a single trial"""

        # Define colors for classes
        class_colors = [
            [255, 102, 178], [102, 158, 255], [153, 51, 255], [255, 51, 51],
            [102, 255, 102], [255, 153, 102], [0, 153, 0], [0, 0, 128],
            [255, 255, 0], [255, 0, 255], [128, 128, 0], [0, 204, 204],
            [255, 102, 178], [255, 165, 0]
        ]
        class_colors = [[r/255, g/255, b/255] for r, g, b in class_colors]

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(12, 1)

        # 1. Top features summary (rows 0-2)
        ax1 = fig.add_subplot(gs[0:2, 0])
        feature_importance_avg = np.mean(importance_map, axis=1)
        top_features = np.argsort(feature_importance_avg)[-20:]
        bars = ax1.bar(range(len(top_features)), feature_importance_avg[top_features])
        ax1.set_title(f'Top 20 Most Important Features - {video_name}', fontsize=14)
        ax1.set_ylabel('Average Importance')
        ax1.set_xlabel('Feature Rank')
        ax1.set_xticks(range(len(top_features)))
        ax1.set_xticklabels([f'F{idx}' for idx in top_features], rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. Speed plot if available (rows 2-4)
        ax2 = fig.add_subplot(gs[2:4, 0])
        if feature_tensor.shape[1] > 3:  # Assuming speed is at index 3
            speed = feature_tensor[0, 3, :].detach().cpu().numpy()
            ax2.plot(np.arange(len(speed)), speed, label='Speed', color='blue', linewidth=2, alpha=0.7)
            ax2.set_title('Speed Feature', fontsize=14)
            ax2.set_ylabel('Speed')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Speed feature not available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Speed Feature (N/A)', fontsize=14)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # 3. Ground truth vs predictions (rows 4-5)
        ax3 = fig.add_subplot(gs[4:5, 0], sharex=ax2)
        if gt_label is not None and not np.all(gt_label == -100):
            time_points = np.arange(len(gt_label))
            ax3.plot(time_points, gt_label, label='Ground Truth', linewidth=2, alpha=0.8, color='black')
            ax3.plot(time_points, predictions, label='Predictions', linewidth=2, alpha=0.7, color='red')
            ax3.set_title('Ground Truth vs Predictions', fontsize=14)
            ax3.set_ylabel('Class')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Ground truth not available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Ground Truth vs Predictions (GT N/A)', fontsize=14)
        plt.setp(ax3.get_xticklabels(), visible=False)

        # 4. Class attention scores (rows 5-8)
        ax4 = fig.add_subplot(gs[5:8, 0], sharex=ax2)
        for i, (class_id, attention) in enumerate(class_attention_maps.items()):
            color_idx = i % len(class_colors)
            class_name = self.event_list[class_id] if class_id < len(self.event_list) else f'Class_{class_id}'
            ax4.plot(np.arange(len(attention)), attention,
                    label=f'{class_name} (ID:{class_id})',
                    color=class_colors[color_idx], linewidth=2)
        ax4.set_title('Class-Specific Attention Scores', fontsize=14)
        ax4.set_ylabel('Attention Score')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.get_xticklabels(), visible=False)

        # 5. Feature importance heatmap (rows 8-12)
        ax5 = fig.add_subplot(gs[8:12, 0], sharex=ax2)

        # Limit to most relevant features for visualization
        if importance_map.shape[0] > 25:
            first_16 = np.arange(16)
            top_9 = [idx for idx in top_features[::-1] if idx not in first_16][:9]
            selected_features = np.concatenate([first_16, top_9])
            display_map = importance_map[selected_features, :]
        else:
            selected_features = np.arange(importance_map.shape[0])
            display_map = importance_map

        im = ax5.imshow(display_map, aspect='auto', cmap='hot', interpolation='nearest',
                       extent=[0, display_map.shape[1], -0.5, len(selected_features)-0.5])

        # Add feature group rectangles if applicable
        row_groups = [(0, 4), (5, 8), (9, 15), (16, 25)]
        for start, end in row_groups:
            if start < len(selected_features):
                rect_end = min(end, len(selected_features)-1)
                rect = Rectangle(
                    (-0.5, start - 0.5), display_map.shape[1], rect_end - start + 1,
                    linewidth=2, edgecolor='cyan', facecolor='none'
                )
                ax5.add_patch(rect)

        ax5.set_title(f'Feature Importance Heatmap - {video_name}', fontsize=14)
        ax5.set_ylabel('Feature Index')
        ax5.set_xlabel('Time')
        ax5.set_yticks(np.arange(len(selected_features)))
        ax5.set_yticklabels([str(idx) for idx in selected_features])

        # Align x-axes
        n_time = display_map.shape[1]
        xticks = np.linspace(0, n_time-1, min(11, n_time), dtype=int)
        for ax in [ax2, ax3, ax4, ax5]:
            ax.set_xlim([0, n_time])
            ax.set_xticks(xticks)
        ax5.set_xticklabels([str(x) for x in xticks])

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.close(fig)

    def analyze_multiple_trials(self, test_dataset, trial_indices=None, save_dir=None):
        """Analyze feature importance across multiple trials"""
        if trial_indices is None:
            trial_indices = range(len(test_dataset))

        all_results = []

        print(f"Analyzing {len(trial_indices)} trials...")
        for i, trial_idx in enumerate(tqdm(trial_indices)):
            try:
                results = self.analyze_trial_importance(
                    test_dataset, trial_idx, save_dir
                )
                all_results.append(results)
            except Exception as e:
                print(f"Error analyzing trial {trial_idx}: {e}")
                continue

        # Aggregate results across trials
        aggregated = self._aggregate_trial_results(all_results)

        if save_dir:
            # Save aggregated results
            np.save(os.path.join(save_dir, 'aggregated_feature_importance.npy'), aggregated)
            self._create_aggregate_visualization(aggregated, save_dir)

        return all_results, aggregated

    def _aggregate_trial_results(self, all_results):
        """Aggregate feature importance across multiple trials"""
        if not all_results:
            return {}

        # Collect all importance maps
        importance_maps = [r['importance_map'] for r in all_results]

        # Average importance across trials
        avg_importance = np.mean(importance_maps, axis=0)
        std_importance = np.std(importance_maps, axis=0)

        # Collect top features per class across trials
        class_feature_counts = {}
        for results in all_results:
            for class_id, features in results['top_features_per_class'].items():
                if class_id not in class_feature_counts:
                    class_feature_counts[class_id] = {}

                for feat_idx in features['indices']:
                    if feat_idx not in class_feature_counts[class_id]:
                        class_feature_counts[class_id][feat_idx] = 0
                    class_feature_counts[class_id][feat_idx] += 1

        return {
            'average_importance': avg_importance,
            'std_importance': std_importance,
            'class_feature_frequency': class_feature_counts,
            'num_trials': len(all_results),
            'trial_names': [r['video_name'] for r in all_results]
        }

    def _create_aggregate_visualization(self, aggregated, save_dir):
        """Create visualization of aggregated results across trials"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Average importance heatmap
        ax1 = axes[0, 0]
        im1 = ax1.imshow(aggregated['average_importance'][:20], aspect='auto', cmap='hot')
        ax1.set_title(f"Average Feature Importance\n({aggregated['num_trials']} trials)")
        ax1.set_ylabel('Feature Index')
        ax1.set_xlabel('Time')
        plt.colorbar(im1, ax=ax1)

        # Standard deviation heatmap
        ax2 = axes[0, 1]
        im2 = ax2.imshow(aggregated['std_importance'][:20], aspect='auto', cmap='viridis')
        ax2.set_title("Feature Importance Std Dev")
        ax2.set_ylabel('Feature Index')
        ax2.set_xlabel('Time')
        plt.colorbar(im2, ax=ax2)

        # Overall feature ranking
        ax3 = axes[1, 0]
        overall_importance = np.mean(aggregated['average_importance'], axis=1)
        top_20 = np.argsort(overall_importance)[-20:]
        ax3.bar(range(len(top_20)), overall_importance[top_20])
        ax3.set_title('Top 20 Features (Averaged Across Time & Trials)')
        ax3.set_ylabel('Average Importance')
        ax3.set_xlabel('Feature Rank')
        ax3.set_xticks(range(len(top_20)))
        ax3.set_xticklabels([f'F{idx}' for idx in top_20], rotation=45)

        # Class-specific feature frequency
        ax4 = axes[1, 1]
        if aggregated['class_feature_frequency']:
            # Show most frequently important features per class
            class_data = []
            labels = []

            for class_id, feat_counts in aggregated['class_feature_frequency'].items():
                if feat_counts:
                    most_freq_feat = max(feat_counts.items(), key=lambda x: x[1])
                    class_data.append(most_freq_feat[1] / aggregated['num_trials'] * 100)
                    class_name = self.event_list[class_id] if class_id < len(self.event_list) else f'C{class_id}'
                    labels.append(f"{class_name}\n(F{most_freq_feat[0]})")

            if class_data:
                ax4.bar(range(len(class_data)), class_data)
                ax4.set_title('Most Frequently Important Feature per Class\n(% of trials)')
                ax4.set_ylabel('Frequency (%)')
                ax4.set_xlabel('Class (Top Feature)')
                ax4.set_xticks(range(len(labels)))
                ax4.set_xticklabels(labels, rotation=45, ha='right')

        if not aggregated['class_feature_frequency'] or not any(aggregated['class_feature_frequency'].values()):
            ax4.text(0.5, 0.5, 'No class-specific\ndata available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Class-Specific Feature Frequency')

        plt.tight_layout()

        save_path = os.path.join(save_dir, 'aggregated_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Aggregated analysis saved to {save_path}")
        plt.close(fig)

    def cleanup(self):
        """Clean up resources"""
        self.grad_cam.cleanup()


def main():
    """
    Enhanced main function that integrates with MovFormer's trial-based system
    """
    parser = argparse.ArgumentParser(
        description="Analyze feature importance for MovFormer model using integrated system"
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Config file used for training')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--trial_idx', type=int, default=None,
                       help='Specific trial index to analyze')
    parser.add_argument('--analyze_all', action='store_true',
                       help='Analyze all trials and create aggregate results')
    parser.add_argument('--max_trials', type=int, default=10,
                       help='Maximum number of trials to analyze (when analyze_all=True)')
    parser.add_argument('--output_dir', type=str, default='feature_importance_results',
                       help='Directory to save results')
    parser.add_argument('--target_layer', type=str, default='encoder.encoder.layers.7',
                       help='Layer to analyze for Grad-CAM')

    args = parser.parse_args()

    # Load configuration
    all_params = load_config_file(args.config)

    # Set device
    if args.device != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_params['device'] = device

    print(f"Using device: {device}")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")

    # Load dataset using existing system (same as main2.py)
    test_nc_paths = all_params["test_nc_paths"]
    test_trial_dict, test_trial_names = get_trial_dict(all_params, test_nc_paths)
    test_data_dict, feature_dim = get_data_dict(
        all_params=all_params,
        nc_paths=test_nc_paths,
        trial_dict=test_trial_dict,
        trial_names=test_trial_names,
    )

    # Load event mapping
    event_list = np.loadtxt(all_params["mapping_file"], dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)

    # Update encoder params with input_dim
    all_params['encoder_params']['input_dim'] = feature_dim
    print(f"Feature dimension: {feature_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Event classes: {event_list}")

    # Create test dataset
    test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')
    print(f"Test dataset size: {len(test_test_dataset)}")

    # Initialize trainer and load model (reuse Trainer from main2.py)
    from main2 import Trainer
    trainer = Trainer(all_params, event_list)
    trainer.model.load_state_dict(torch.load(args.model_path, weights_only=True))
    trainer.model.to(device)
    trainer.model.eval()

    print("Model loaded successfully")

    # Initialize feature importance analyzer
    analyzer = TrialFeatureImportanceAnalyzer(
        trainer, device, args.target_layer
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.trial_idx is not None:
            # Analyze specific trial
            print(f"Analyzing single trial: {args.trial_idx}")

            if args.trial_idx >= len(test_test_dataset):
                print(f"ERROR: Trial index {args.trial_idx} out of range (0-{len(test_test_dataset)-1})")
                return 1

            results = analyzer.analyze_trial_importance(
                test_test_dataset, args.trial_idx, args.output_dir
            )

            # Print summary
            print(f"\n=== RESULTS FOR TRIAL {args.trial_idx} ===")
            print(f"Video: {results['video_name']}")
            print(f"NetCDF: {results['nc_path']}")
            print(f"Predicted classes: {results['predicted_classes']}")

            print(f"\nTop features per class:")
            for class_id, feat_info in results['top_features_per_class'].items():
                class_name = feat_info['class_name']
                top_3_features = feat_info['indices'][:3]
                top_3_scores = feat_info['scores'][:3]
                print(f"  {class_name} (ID:{class_id}):")
                for i, (feat_idx, score) in enumerate(zip(top_3_features, top_3_scores)):
                    print(f"    {i+1}. Feature {feat_idx}: {score:.4f}")

        elif args.analyze_all:
            # Analyze all trials (limited by max_trials)
            print(f"Analyzing multiple trials (max: {args.max_trials})")

            trial_indices = list(range(min(args.max_trials, len(test_test_dataset))))

            all_results, aggregated = analyzer.analyze_multiple_trials(
                test_test_dataset, trial_indices, args.output_dir
            )

            print(f"\n=== AGGREGATED RESULTS ===")
            print(f"Successfully analyzed {len(all_results)} trials")
            print(f"Results saved to: {args.output_dir}")

            if aggregated and 'average_importance' in aggregated:
                # Show overall most important features
                overall_importance = np.mean(aggregated['average_importance'], axis=1)
                top_10_overall = np.argsort(overall_importance)[-10:][::-1]

                print(f"\nTop 10 most important features overall:")
                for i, feat_idx in enumerate(top_10_overall):
                    print(f"  {i+1}. Feature {feat_idx}: {overall_importance[feat_idx]:.4f}")

        else:
            print("ERROR: Must specify either --trial_idx or --analyze_all")
            print("Use --trial_idx N to analyze a specific trial")
            print("Use --analyze_all to analyze multiple trials")
            return 1

        print(f"\nAnalysis complete! Results saved to: {args.output_dir}")

    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Clean up
        analyzer.cleanup()

    return 0


if __name__ == '__main__':
    main()
