#!/usr/bin/env python3
"""
Example script demonstrating how to use the integrated feature importance analysis
with the MovFormer system.

This script shows how to:
1. Use the TrialFeatureImportanceAnalyzer directly in your code
2. Integrate with existing Trainer and dataset loading
3. Analyze feature importance for specific trials
"""

import os
import torch
import numpy as np
from pathlib import Path

# Import the necessary components
from main2 import Trainer
from moveseg.utils.dataset import VideoFeatureDataset, get_trial_dict, get_data_dict
from moveseg.diffact.utils import load_config_file
from moveseg.features.interpretability import TrialFeatureImportanceAnalyzer


def example_single_trial_analysis():
    """
    Example: Analyze feature importance for a single trial
    """
    print("=== Single Trial Feature Importance Analysis ===\n")

    # Configuration (adjust these paths to your setup)
    config_path = r"D:\Akseli\Code\MovFormer\result\Freddy_train_20251009_longerFewS3D\all_params.json"  # Replace with your config
    model_path = r"D:\Akseli\Code\MovFormer\result\Freddy_train_20251009_longerFewS3D\epoch-75.model"  # Replace with your trained model
    output_dir = r"D:\Akseli\Code\MovFormer\feature_importance_results"
    trial_idx = 0  # Which trial to analyze

    # Load configuration
    all_params = load_config_file(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_params['device'] = device

    # Load test dataset (same as main2.py)
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

    # Update encoder params
    all_params['encoder_params']['input_dim'] = feature_dim

    # Create test dataset
    test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    print(f"Dataset loaded: {len(test_dataset)} trials")
    print(f"Feature dimension: {feature_dim}")
    print(f"Classes: {event_list}")

    # Initialize trainer and load model
    trainer = Trainer(all_params, event_list)
    trainer.model.load_state_dict(torch.load(model_path, weights_only=True))
    trainer.model.to(device)
    trainer.model.eval()

    print(f"Model loaded from: {model_path}")

    # Initialize feature importance analyzer
    analyzer = TrialFeatureImportanceAnalyzer(trainer, device)

    # Analyze specific trial
    results = analyzer.analyze_trial_importance(
        test_dataset, trial_idx, save_dir=output_dir
    )

    # Print results
    print(f"\n--- Results for Trial {trial_idx} ---")
    print(f"Video: {results['video_name']}")
    print(f"Predicted classes: {results['predicted_classes']}")

    print(f"\nTop 5 most important features per predicted class:")
    for class_id, feat_info in results['top_features_per_class'].items():
        class_name = feat_info['class_name']
        top_5_features = feat_info['indices'][:5]
        top_5_scores = feat_info['scores'][:5]

        print(f"\n{class_name} (Class ID: {class_id}):")
        for i, (feat_idx, score) in enumerate(zip(top_5_features, top_5_scores)):
            print(f"  {i+1}. Feature {feat_idx}: {score:.4f}")

    # Get overall most important features (averaged over time)
    overall_importance = np.mean(results['importance_map'], axis=1)
    top_10_overall = np.argsort(overall_importance)[-10:][::-1]

    print(f"\nTop 10 most important features overall:")
    for i, feat_idx in enumerate(top_10_overall):
        print(f"  {i+1}. Feature {feat_idx}: {overall_importance[feat_idx]:.4f}")

    print(f"\nVisualization saved to: {output_dir}")

    # Cleanup
    analyzer.cleanup()


def example_multiple_trials_analysis():
    """
    Example: Analyze feature importance across multiple trials
    """
    print("\n=== Multiple Trials Feature Importance Analysis ===\n")

    # Configuration (adjust these paths to your setup)
    config_path = "configs/your_config.json"  # Replace with your config
    model_path = "result/your_model/epoch-300.model"  # Replace with your trained model
    output_dir = "feature_importance_multi_trial"
    max_trials = 5  # Analyze first 5 trials

    # Setup (same as above)
    all_params = load_config_file(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_params['device'] = device

    test_nc_paths = all_params["test_nc_paths"]
    test_trial_dict, test_trial_names = get_trial_dict(all_params, test_nc_paths)
    test_data_dict, feature_dim = get_data_dict(
        all_params=all_params, nc_paths=test_nc_paths,
        trial_dict=test_trial_dict, trial_names=test_trial_names,
    )

    event_list = np.loadtxt(all_params["mapping_file"], dtype=str)
    event_list = [i[1] for i in event_list]
    num_classes = len(event_list)
    all_params['encoder_params']['input_dim'] = feature_dim

    test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    trainer = Trainer(all_params, event_list)
    trainer.model.load_state_dict(torch.load(model_path, weights_only=True))
    trainer.model.to(device)
    trainer.model.eval()

    analyzer = TrialFeatureImportanceAnalyzer(trainer, device)

    # Analyze multiple trials
    trial_indices = list(range(min(max_trials, len(test_dataset))))
    all_results, aggregated = analyzer.analyze_multiple_trials(
        test_dataset, trial_indices, save_dir=output_dir
    )

    print(f"Analyzed {len(all_results)} trials successfully")

    if aggregated and 'average_importance' in aggregated:
        # Show aggregate results
        overall_importance = np.mean(aggregated['average_importance'], axis=1)
        top_10_overall = np.argsort(overall_importance)[-10:][::-1]

        print(f"\nTop 10 most consistently important features:")
        for i, feat_idx in enumerate(top_10_overall):
            print(f"  {i+1}. Feature {feat_idx}: {overall_importance[feat_idx]:.4f}")

        # Show class-specific frequently important features
        print(f"\nMost frequently important features per class:")
        for class_id, feat_counts in aggregated['class_feature_frequency'].items():
            if feat_counts:
                most_freq_feat = max(feat_counts.items(), key=lambda x: x[1])
                freq_percentage = most_freq_feat[1] / aggregated['num_trials'] * 100
                class_name = event_list[class_id] if class_id < len(event_list) else f'Class_{class_id}'
                print(f"  {class_name}: Feature {most_freq_feat[0]} ({freq_percentage:.1f}% of trials)")

    print(f"\nAggregate results and visualizations saved to: {output_dir}")

    # Cleanup
    analyzer.cleanup()


def example_programmatic_usage():
    """
    Example: Use the analyzer programmatically without saving files
    """
    print("\n=== Programmatic Usage Example ===\n")

    # This is a minimal example showing how to extract just the data
    # without generating plots or saving files

    # ... (setup same as above - config, dataset, model loading)
    # For brevity, I'll show just the analysis part:

    # analyzer = TrialFeatureImportanceAnalyzer(trainer, device)
    #
    # # Get raw importance data without saving plots
    # results = analyzer.analyze_trial_importance(
    #     test_dataset, trial_idx=0, save_dir=None  # No saving
    # )
    #
    # # Extract what you need:
    # importance_map = results['importance_map']  # Shape: [F, T]
    # class_attention = results['class_attention_maps']  # {class_id: [T]}
    # predictions = results['predictions']  # [T]
    # top_features = results['top_features_per_class']
    #
    # # Use the data in your own analysis...
    # # For example, correlate with your own behavioral annotations
    #
    # print("Raw feature importance data extracted successfully!")

    print("See the code above for programmatic usage pattern")


if __name__ == "__main__":

    # Uncomment to run:
    example_single_trial_analysis()
    # example_multiple_trials_analysis()
    # example_programmatic_usage()
