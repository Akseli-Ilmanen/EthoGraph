import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import pickle
style_path = Path(__file__).resolve().parents[2] / 'configs' / 'style.mplstyle'
plt.style.use(str(style_path))
F1_COLOURS = ["#9467bd", "#2ecc71", "#e377c2"]  # purple, green, pink
PRED_COLOURS = ["#1f77b4", "#d62728"]  # uncorrected: blue, corrected: red

    
def plot_metrics_best_model(results_dir: Path, result_paths: List[str], 
                           f1_thresholds: List[float], fps: float, epoch: int) -> None:
    nested_results_list = []
    for path in result_paths:
        
        path = os.path.join(path, f"test_results_epoch{epoch}.npy")        
        data = np.load(path, allow_pickle=True).item()
        nested_results_list.append(data)
    
    fig, axs = plt.subplot_mosaic([
        ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
        ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
        ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
        ['D', 'D', 'D', 'D', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['D', 'D', 'D', 'D', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['D', 'D', 'D', 'D', 'F', 'F', 'F', 'F', 'F', 'F'],
        ['E', 'E', 'E', 'E', 'G', 'G', 'G', 'G', 'G', 'G'],
        ['E', 'E', 'E', 'E', 'G', 'G', 'G', 'G', 'G', 'G'],
        ['E', 'E', 'E', 'E', 'G', 'G', 'G', 'G', 'G', 'G']
    ], figsize=(20, 15))

    # A: IoU visualization
    plot_iou_visualization(axs['A'], ious=f1_thresholds)

    # B: Overall metrics comparison
    plot_overall_metrics_comparison(axs['B'], nested_results_list, f1_thresholds)

    # # C: Metrics over epochs (if available)
    # plot_metrics_over_epochs_on_axis(axs['C'], result_paths, f1_thresholds)

    # D: IOU hist + TP, FP, FN comparison
    plot_iou_distribution_with_tp_fp_fn(axs['D'], nested_results_list, f1_thresholds)

    # E: Combined temporal deltas histogram
    plot_combined_deltas_histogram(axs['E'], nested_results_list, fps)

    # F: Labelwise results for uncorrected
    uncorrected_classwise = [nr["uncorrected"]["classwise_results"] for nr in nested_results_list]
    plot_labelwise_results(axs['F'], uncorrected_classwise, f1_thresholds, title="Uncorrected F1 Scores")

    # G: Labelwise results for corrected
    corrected_classwise = [nr["corrected"]["classwise_results"] for nr in nested_results_list]
    plot_labelwise_results(axs['G'], corrected_classwise, f1_thresholds, title="Corrected F1 Scores")

    plt.tight_layout()
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_png = os.path.join(results_dir, f"eval_summary_epoch{epoch}_{dt_str}.png")
    path_pdf = os.path.join(results_dir, f"eval_summary_epoch{epoch}_{dt_str}.pdf")
    path_eps = os.path.join(results_dir, f"eval_summary_epoch{epoch}_{dt_str}.eps")
    
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.savefig(path_eps, format='eps', bbox_inches='tight')
    # plt.savefig(path_pdf, format='pdf', dpi=600, bbox_inches='tight', pad_inches=0.05, facecolor='white', edgecolor='none')
    # plt.savefig(path, 
    #             format='pdf',
    #             dpi=600,  # Increased DPI for rasterized elements
    #             bbox_inches='tight',
    #             pad_inches=0.05,
    #             facecolor='white',
    #             edgecolor='none')
    plt.close(fig)


def plot_overall_metrics_comparison(ax, nested_results_list: List[Dict[str, Any]], 
                                   f1_thresholds: List[float]) -> None:
    """Plot comparison bar chart of overall metrics with mean bars and individual points."""
    if 'Frame_F1' in nested_results_list[0]['uncorrected']:
        metrics_to_plot = ['Acc', 'Frame_F1'] + [f'F1@{int(th*100)}' for th in f1_thresholds]
        metric_labels = ['Accuracy', 'F1-Framewise'] + [f'F1@{int(th*100)}' for th in f1_thresholds]
    else:
        metrics_to_plot = ['Acc'] +  [f'F1@{int(th*100)}' for th in f1_thresholds]
        metric_labels = ['Accuracy'] + [f'F1@{int(th*100)}' for th in f1_thresholds]

    # Calculate means and collect individual values
    uncorrected_means = []
    corrected_means = []
    uncorrected_points = []
    corrected_points = []
    
    for metric in metrics_to_plot:
        unc_vals = [nr['uncorrected'][metric] for nr in nested_results_list]
        corr_vals = [nr['corrected'][metric] for nr in nested_results_list]
        uncorrected_means.append(np.mean(unc_vals))
        corrected_means.append(np.mean(corr_vals))
        uncorrected_points.append(unc_vals)
        corrected_points.append(corr_vals)

    x = range(len(metrics_to_plot))
    width = 0.35

    # Plot mean bars
    ax.bar([i - width/2 for i in x], uncorrected_means, width, label='Uncorrected', 
           alpha=0.8, color=PRED_COLOURS[0])
    ax.bar([i + width/2 for i in x], corrected_means, width, label='Corrected', 
           alpha=0.8, color=PRED_COLOURS[1])

    # Plot individual points
    for i in x:
        ax.scatter([i - width/2] * len(uncorrected_points[i]), uncorrected_points[i],
                  color=PRED_COLOURS[0], s=20, alpha=0.6, zorder=3)
        ax.scatter([i + width/2] * len(corrected_points[i]), corrected_points[i],
                  color=PRED_COLOURS[1], s=20, alpha=0.6, zorder=3)

    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)

    # Color the F1 threshold labels with their corresponding F1 colors
    colored_labels = []
    for i, label in enumerate(metric_labels):
        if label.startswith('F1@'):
            # Find which F1 threshold this corresponds to
            f1_idx = i - 2  # First two are Accuracy and Edit Score
            colored_labels.append(label)
        else:
            colored_labels.append(label)

    ax.set_xticklabels(colored_labels)



    # Add value labels on bars (showing means)
    for i, (unc_val, corr_val) in enumerate(zip(uncorrected_means, corrected_means)):
        ax.text(i - width/2, unc_val + 0.01, f'{unc_val:.3f}', ha='center', va='bottom')
        ax.text(i + width/2, corr_val + 0.01, f'{corr_val:.3f}', ha='center', va='bottom')


def plot_iou_visualization(ax, ious=[0.5, 0.75, 0.9]):
    gt_len = 30.0
    pred_len = 30.0
    left_bg = 20.0
    right_bg = 20.0

    total_len = int(left_bg + gt_len + right_bg)
    gt_start = left_bg
    gt_end = gt_start + gt_len

    heights = [0.4, 1.1, 1.8, 2.5]

    for i, iou in enumerate(ious):
        I = (gt_len + pred_len) * iou / (1.0 + iou)

        pred_start = gt_start + (gt_len - I)
        pred_end = pred_start + pred_len

        ax.set_xlim(-1, total_len + 1)
        ax.set_ylim(0, 3)
        ax.set_yticks([])

        if i == 0:
            ax.barh(heights[-1], gt_len, left=gt_start, height=0.6, color='gray', label='Ground Truth')
            ax.hlines(heights[-1], 0, total_len, linewidth=6, color='lightgray')
            ax.vlines(gt_start, 0, heights[-1], color='gray', linestyles="dotted")
            ax.vlines(gt_end, 0, heights[-1], color='gray', linestyles="dotted")

        ax.barh(heights[i], pred_len, left=pred_start, height=0.6, color=F1_COLOURS[i])
        ax.text(10, heights[i]+0.1, f'IoU = {iou}',
                ha='center', va='center', fontweight='bold')
        ax.hlines(heights[i], 0, total_len, linewidth=6, color='lightgray')

    ax.set_xlabel('Samples')

    legend_elements = [
        Patch(facecolor='gray', label='Ground Truth')
    ]

    for i, th in enumerate(ious):
        legend_elements.append(Patch(facecolor=F1_COLOURS[i], label=f'F1@{int(th*100)}'))

    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)

    return legend_elements



def create_broken_histogram(data, break_range, bins=50, labels=None,  ax=None, ylog=False, xlabel="Values"):
    """Create histogram with broken x-axis, supports multiple datasets
    
    ax = (ax1, ax2)
    """
    d = 0.015 

    if not isinstance(data, list):
        data = [data]
    

    ax1, ax2 = ax
    
    for i, (dataset, label) in enumerate(zip(data, labels)):
        ax1.hist(dataset, bins=bins, alpha=0.7, label=label)
        ax2.hist(dataset, bins=bins, alpha=0.7, label=label)
    
    all_data = np.concatenate(data)
    ax1.set_xlim(0, break_range[0])
    ax2.set_xlim(break_range[1], all_data.max() * 1.05)
    
    ax2.tick_params(axis='y', left=False, labelleft=False)
    
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

    ax1.set_ylabel('Count (log scale)')
    ax1.set_xlabel(xlabel)

    if ylog:
        ax1.set_yscale('log')
        ax2.set_yscale('log')


    ax2.legend()
    
    # Avoid overlap in xtick nums
    xticks = ax2.get_xticks()
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([""] + [f"{t:g}" for t in xticks[1:]])

    return (ax1, ax2)



def plot_labelwise_results(ax, results_list: List[Dict], f1_thresholds: List[float], title: str) -> None:
    """
    Create nested bar plot for F1 scores with mean bars and individual points.
    
    Args:
        ax: Matplotlib axis to plot on
        results_list: List of dictionaries returned by func_eval_labelwise
        f1_thresholds: List of F1 threshold values
    """
    # Get classes from first result (assuming all have same classes)
    classes = [int(cls) for cls in results_list[0].keys()]
    n_classes = len(classes)
    
    # Calculate means and collect individual values
    f1_means = np.zeros((n_classes, 3))
    f1_points = [[] for _ in range(n_classes)]
    
    for cls_idx, cls in enumerate(classes):
        for result in results_list:
            f1_points[cls_idx].append(result[cls]['f1s'])
        f1_means[cls_idx] = np.mean(f1_points[cls_idx], axis=0)
    
    x = np.arange(n_classes)
    width = 0.2
    
    # Plot mean bars
    bars1 = ax.bar(x - 0.5*width, f1_means[:, 0], width, 
                   label=f'F1@{int(f1_thresholds[0]*100)}', alpha=0.8, color=F1_COLOURS[0])
    bars2 = ax.bar(x + 0.5*width, f1_means[:, 1], width, 
                   label=f'F1@{int(f1_thresholds[1]*100)}', alpha=0.8, color=F1_COLOURS[1])
    bars3 = ax.bar(x + 1.5*width, f1_means[:, 2], width, 
                   label=f'F1@{int(f1_thresholds[2]*100)}', alpha=0.8, color=F1_COLOURS[2])

    # Plot individual points
    for cls_idx in range(n_classes):
        for point in f1_points[cls_idx]:
            ax.scatter(x[cls_idx] - 0.5*width, point[0], color=F1_COLOURS[0], s=15, alpha=0.6, zorder=3)
            ax.scatter(x[cls_idx] + 0.5*width, point[1], color=F1_COLOURS[1], s=15, alpha=0.6, zorder=3)
            ax.scatter(x[cls_idx] + 1.5*width, point[2], color=F1_COLOURS[2], s=15, alpha=0.6, zorder=3)
    
    ax.set_xlabel('Motifs')
    ax.set_xticks(x)
    
    
    ax.set_ylabel('F1 Score (%)')
    
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_title(title)


def plot_metrics_over_epochs_on_axis(
    ax, 
    result_paths: List[Path], 
    f1_thresholds: List[float]
) -> None:
    all_result_files = []
    for results_dir in result_paths:
        folder = Path(results_dir)
        result_files = sorted(
            folder.glob("test_results_epoch*.npy"),
            key=lambda x: int(x.stem.split("epoch")[-1])
        )
        all_result_files.extend(result_files)
    
    if not all_result_files:
        ax.text(0.5, 0.5, 'No epoch data available', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    all_epochs = []
    all_corrected_metrics = []
    
    for file_path in all_result_files:
        try:
            epoch = int(file_path.stem.split("epoch")[-1])
            try:
                nested_data = np.load(file_path, allow_pickle=True).item()
            except Exception as e:
                print(f"Skipping corrupted file {file_path.name}: {type(e).__name__}")
                continue
            
            data = nested_data["uncorrected"]
            metrics = {"Acc": float(data["Acc"])}
            for threshold in f1_thresholds:
                key = f"F1@{int(threshold*100)}"
                metrics[key] = float(data[key])
            
            all_epochs.append(epoch)
            all_corrected_metrics.append(metrics)
            
        except (pickle.UnpicklingError, ValueError, KeyError, OSError) as e:
            print(f"Skipping {file_path.name}: {type(e).__name__}")
            continue
    
    if not all_epochs:
        ax.text(0.5, 0.5, 'No valid epoch data found', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    unique_epochs = sorted(set(all_epochs))
    metric_keys = ["Acc"] + [f"F1@{int(th*100)}" for th in f1_thresholds]
    mean_metrics = {key: [] for key in metric_keys}
    min_metrics = {key: [] for key in metric_keys}
    max_metrics = {key: [] for key in metric_keys}
    
    for epoch in unique_epochs:
        epoch_data = [m for e, m in zip(all_epochs, all_corrected_metrics) if e == epoch]
        for key in metric_keys:
            values = [d[key] for d in epoch_data]
            mean_metrics[key].append(np.mean(values))
            min_metrics[key].append(np.min(values))
            max_metrics[key].append(np.max(values))
    
    ax.plot(unique_epochs, mean_metrics["Acc"], 
            label="Accuracy", marker="o", color='black')
    ax.fill_between(unique_epochs, min_metrics["Acc"], max_metrics["Acc"], 
                    color='black', alpha=0.2)
    
    for i, threshold in enumerate(f1_thresholds):
        key = f"F1@{int(threshold*100)}"
        ax.plot(unique_epochs, mean_metrics[key], 
                label=key, marker="s", color=F1_COLOURS[i])
        ax.fill_between(unique_epochs, min_metrics[key], max_metrics[key], 
                       color=F1_COLOURS[i], alpha=0.2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Metrics over Epochs (Uncorrected)")
    ax.legend()

def plot_combined_deltas_histogram(ax, nested_results_list: List[Dict[str, Any]], fps: float) -> None:
    # Concatenate all deltas
    all_uncorrected_start = np.concatenate([nr["uncorrected"]["start_deltas"] 
                                           for nr in nested_results_list])
    all_uncorrected_end = np.concatenate([nr["uncorrected"]["end_deltas"] 
                                         for nr in nested_results_list])
    all_corrected_start = np.concatenate([nr["corrected"]["start_deltas"] 
                                         for nr in nested_results_list])
    all_corrected_end = np.concatenate([nr["corrected"]["end_deltas"] 
                                       for nr in nested_results_list])
    
    uncorrected_deltas = np.concatenate([all_uncorrected_start, all_uncorrected_end]) / fps
    corrected_deltas = np.concatenate([all_corrected_start, all_corrected_end]) / fps

    ax.hist(uncorrected_deltas, bins=50, alpha=0.4, label='Uncorrected', color=PRED_COLOURS[0])
    ax.hist(corrected_deltas, bins=50, alpha=0.4, label='Corrected', color=PRED_COLOURS[1])

    ax.set_yscale('log')
    ax.set_xlabel('Temporal Deltas (s)')
    ax.set_ylabel('Count (log scale)')
    ax.legend()


def plot_iou_distribution_with_tp_fp_fn(
    ax, 
    nested_results_list: List[Dict[str, Any]],
    f1_thresholds: List[float]
) -> None:
    all_uncorrected_IoUs = np.concatenate([nr["uncorrected"]["all_IoUs"]
                                           for nr in nested_results_list])
    all_corrected_IoUs = np.concatenate([nr["corrected"]["all_IoUs"]
                                         for nr in nested_results_list])
    
    ax.hist(all_uncorrected_IoUs, bins=50, alpha=0.4, 
            label='Uncorrected', color=PRED_COLOURS[0])
    ax.hist(all_corrected_IoUs, bins=50, alpha=0.4, 
            label='Corrected', color=PRED_COLOURS[1])
    
    for i, th in enumerate(f1_thresholds):
        ax.axvline(x=th, color=F1_COLOURS[i], linestyle='--')
    
    ax.set_yscale('log')
    ax.set_xlabel('IoU')
    ax.set_ylabel('Count (log scale)')
    ax.legend(loc='upper right')
    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    axins = inset_axes(
        ax,
        width="30%",
        height="30%",
        loc='upper left',
        borderpad=3
    )
    
    metrics = ['FP', 'FN', 'TP']
    corrected_means = []
    uncorrected_means = []
    
    for metric in metrics:
        corr_vals = [int(nr['corrected'][metric]) for nr in nested_results_list]
        uncorr_vals = [int(nr['uncorrected'][metric]) for nr in nested_results_list]

        corrected_means.append(np.sum(corr_vals))
        uncorrected_means.append(np.sum(uncorr_vals))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axins.bar(x - width/2, uncorrected_means, width,
              label='Uncorrected', alpha=0.8, color=PRED_COLOURS[0])
    axins.bar(x + width/2, corrected_means, width,
              label='Corrected', alpha=0.8, color=PRED_COLOURS[1])
    
    axins.set_xticks(x)
    axins.set_xticklabels(metrics, fontsize=8)
    axins.tick_params(axis='y', labelsize=8)
    axins.set_ylabel('Count', fontsize=8)
    axins.legend(fontsize=7, loc='upper right')
    


