# Utils not from diffact but added for preparing data for training, and doing inference on new data
import numpy as np
from ethograph.utils.labels import get_labels_start_end_time


def func_eval(ground_truth_dict, predictions_dict, video_list, f1_thresholds=[.5, .75, .9]):
    """
    Evaluate predictions against ground truth using numeric class indices.
    
    Args:
        ground_truth_dict: {video_name: np.array of class indices}
        predictions_dict: {video_name: np.array of class indices}
        video_list: List of video names to evaluate
        f1_thresholds: List of IoU thresholds for F1 score calculation
    """
    
    overlap = f1_thresholds
    num_thresholds = len(f1_thresholds)
    tp, fp, fn = np.zeros(num_thresholds), np.zeros(num_thresholds), np.zeros(num_thresholds)
    tp_base, fp_base, fn_base = 0, 0, 0
 
    correct = 0
    total = 0
    edit = 0
    
    # Frame-wise counts for overall F1
    frame_tp = 0
    frame_fp = 0
    frame_fn = 0


    for vid in video_list:
        gt_content = ground_truth_dict[vid].astype(int)  
        pred_content = predictions_dict[vid].astype(int)

        assert(len(gt_content) == len(pred_content))

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == pred_content[i]:
                correct += 1

        for i in range(len(gt_content)):
            if pred_content[i] == gt_content[i] and gt_content[i] != 0:
                frame_tp += 1
            elif pred_content[i] != 0 and gt_content[i] == 0:
                frame_fp += 1
            elif pred_content[i] == 0 and gt_content[i] != 0:
                frame_fn += 1



        # Convert to string format for existing edit_score and f_score functions
        gt_strings = [str(x) for x in gt_content]
        pred_strings = [str(x) for x in pred_content]

        edit += edit_score(pred_strings, gt_strings)
 
        for s in range(len(overlap)):
            tp1, fp1, fn1, _, _, _ = f_score(pred_strings, gt_strings, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
        
        # TP, FP, FN when no overlap at all.
        tp2, fp2, fn2, _, _, _ = f_score(pred_strings, gt_strings, -1.0)
        tp_base += tp2
        fp_base += fp2
        fn_base += fn2
     
    acc = 100 * float(correct) / total
    edit = (1.0 * edit) / len(video_list)
    
    # Calculate segment-based F1 scores
    f1s = np.zeros(num_thresholds, dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
        f1 = 2.0 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1) * 100
        f1s[s] = f1
    
    # Calculate frame-wise F1 score
    frame_f1 = 2 * frame_tp / (2 * frame_tp + frame_fn + frame_fp)
    frame_f1 *= 100

    return acc, edit, f1s, tp_base, fp_base, fn_base, frame_f1


def func_eval_labelwise(ground_truth_dict, predictions_dict, video_list, bg_class=[0.0], f1_thresholds=[.5, .75, .9]):
    """
    Evaluate model performance for each class separately by masking out other classes.
    
    Args:
        ground_truth_dict: Dictionary containing ground truth data
        predictions_dict: Dictionary containing prediction data  
        video_list: List of video names to evaluate
        bg_class: Background class name(s)
        f1_thresholds: List of IoU thresholds for F1 score calculation
    """


    eval_classes = [c for c in np.sort(np.unique(np.concatenate(list(ground_truth_dict.values())))) if c not in bg_class]

    classResults = {}
    
    all_IoUs = np.array([], dtype=float)


    start_deltas = np.array([], dtype=float)
    end_deltas = np.array([], dtype=float)

    for target_class in eval_classes:
        num_thresholds = len(f1_thresholds)
        tp, fp, fn = np.zeros(num_thresholds), np.zeros(num_thresholds), np.zeros(num_thresholds)
        
        
        for vid in video_list:
            gt_content = ground_truth_dict[vid].astype(int)  
            pred_content = predictions_dict[vid].astype(int)  
                
            assert(len(gt_content) == len(pred_content))
            
            # Mask all classes except target_class and background
            # Convert non-target classes to background
            gt_masked = []
            pred_masked = []
            
            for i in range(len(gt_content)):
                gt_masked.append(gt_content[i] if gt_content[i] == target_class else bg_class[0])
                pred_masked.append(pred_content[i] if pred_content[i] == target_class else bg_class[0])

            for s in range(len(f1_thresholds)):
                tp1, fp1, fn1, IoUs, starts_d, ends_d = f_score(pred_masked, gt_masked, f1_thresholds[s], bg_class=bg_class)
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1


                all_IoUs = np.concatenate((all_IoUs, IoUs))

            start_deltas = np.concatenate((start_deltas, starts_d), axis=0)
            end_deltas = np.concatenate((end_deltas, ends_d), axis=0)

        f1s = np.zeros(num_thresholds, dtype=float)
        for s in range(len(f1_thresholds)):
            denom_p = tp[s] + fp[s] # avoid division by zero
            denom_r = tp[s] + fn[s]
            precision = tp[s] / denom_p if denom_p > 0 else 0.0
            recall = tp[s] / denom_r if denom_r > 0 else 0.0
            denom_f1 = precision + recall
            f1 = 2.0 * (precision * recall) / denom_f1 if denom_f1 > 0 else 0.0
            f1 = np.nan_to_num(f1) * 100
            f1s[s] = f1 

        classResults[int(target_class)] = {
            'f1s': f1s,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }
    return classResults, f1_thresholds, all_IoUs, start_deltas, end_deltas




def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i
 
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
     
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
 
    return score

 
def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)
 
def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
 
    tp = 0
    fp = 0
 
    hits = np.zeros(len(y_label))
    IoUs = np.zeros(len(p_label))

    start_deltas = np.array([], dtype=float)
    end_deltas = np.array([], dtype=float)

 
    for j in range(len(p_label)):
        intersection = np.maximum(0, np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start))
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        
    
        IoU = (1.0*intersection / (union + 1e-10))*([p_label[j] == y_label[x] for x in range(len(y_label))])
        
        if len(IoU) == 0:
            fp += 1
            IoUs[j] = 0.0
        else:
            idx = np.array(IoU).argmax() # Get the best scoring segment
            IoUs[j] = IoU[idx]


            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn), IoUs, start_deltas, end_deltas