from pathlib import Path
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from ethograph.utils.paths import get_project_root
from ethograph.utils.labels import load_label_mapping

def create_classification_probabilities_pdf(label_dt, output_path: Union[str, Path],
                                           confidence_threshold: float = 0.95,
                                           segment_confidence_threshold: float = 0.85):
    """
    Create a PDF with classification probabilities plots for all trials in a label datatree.

    Args:
        label_dt: xarray DataTree containing labels and labels_confidence
        output_path: Path where to save the PDF file
        confidence_threshold: Threshold below which to highlight low confidence regions
        segment_confidence_threshold: Threshold for mean confidence within each label segment
    """
    output_path = Path(output_path)

    trial_nums = label_dt.trials
    N = len(trial_nums)

    mapping_path = get_project_root() / "configs" / "mapping.txt"
    label_mappings = load_label_mapping(mapping_path)
    class_colors = [label_mappings[i]['color'] for i in range(len(label_mappings))]
    num_classes = len(label_mappings)
    
    n_cols = 3
    n_rows = (N + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, trial_num in enumerate(trial_nums):
        trial_ds = label_dt.trial(trial_num)

        if 'labels_confidence' not in trial_ds:
            continue

        labels_confidence = trial_ds.labels_confidence.values.squeeze()
        labels = trial_ds.labels.values.squeeze()

        conf_per_class = np.zeros((num_classes, len(labels)))
        valid_mask = (labels >= 0) & (labels < num_classes)
        conf_per_class[labels[valid_mask].astype(int), np.where(valid_mask)[0]] = labels_confidence[valid_mask]

        remaining = 1.0 - labels_confidence[valid_mask]
        for t_idx, (label, rem) in enumerate(zip(labels[valid_mask], remaining)):
            label = int(label)
            conf_per_class[:, np.where(valid_mask)[0][t_idx]] += rem / (num_classes - 1)
            conf_per_class[label, np.where(valid_mask)[0][t_idx]] -= rem / (num_classes - 1)
                        
                        
                        
        ax = axes[idx]
        for i in range(1, len(conf_per_class)):
            if np.any(conf_per_class[i, :] > 0):  # Only plot if there's data
                ax.plot(conf_per_class[i, :], alpha=0.7, color=class_colors[i])


        max_probs = np.max(conf_per_class, axis=0)

        mask_low = max_probs < confidence_threshold
        ax.scatter(np.where(mask_low)[0], max_probs[mask_low],
                  color='black', s=10, label='Low Confidence', alpha=0.5)

        first = np.argmax(labels != 0)
        last = len(labels) - 1 - np.argmax((labels != 0)[::-1])
    
        confidence = np.mean(labels_confidence[first:last+1])


        
        segment_boundaries = np.concatenate(([0], np.where(np.diff(labels) != 0)[0] + 1, [len(labels)]))
        has_low_segment = False
        
        
        for start, end in zip(segment_boundaries[:-1], segment_boundaries[1:]):
            segment_label = labels[start]
            if segment_label > 0:
                segment_conf = np.mean(max_probs[start:end])
                if segment_conf < segment_confidence_threshold:
                    has_low_segment = True
                    ax.axvspan(start, end, color='red', alpha=0.3, zorder=5)
                    
            

        trial_name = f"trial-{trial_num}"
        ax.set_title(f'{trial_name}\nMean confidence: {confidence:.3f}', fontsize=10)

        label_dt.trial(trial_num).attrs['mean_model_confidence'] = float(confidence)
        
        
        if confidence < confidence_threshold or has_low_segment:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
            ax.set_title(f'{trial_name}\nConfidence: {confidence:.3f}',
                        fontsize=10, color='red', weight='bold')
            label_dt.trial(trial_num).attrs['model_confidence'] = 'low'
        else:
            label_dt.trial(trial_num).attrs['model_confidence'] = 'high'
            
        

        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])

    for j in range(N, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return label_dt