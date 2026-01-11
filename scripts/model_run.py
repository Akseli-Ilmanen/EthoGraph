import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
import matplotlib
matplotlib.use('Agg')


import gc
import json
import random
import torch
import argparse
import numpy as np
from datetime import datetime


from moveseg.utils.dataset import get_trial_dict, get_data_dict, write_bundle_list, save_config
from moveseg.eval.eval_plotting import plot_metrics_best_model
from moveseg.utils.io import TrialTree
from moveseg.utils.paths import get_project_root
from moveseg.utils.labels import load_mapping
from moveseg.model.cetnet_encoder import *
from moveseg.utils.batch_gen import BatchGenerator


if not torch.cuda.is_available():
    raise EnvironmentError("CUDA not available. Please check your PyTorch installation.")
gc.collect()
torch.cuda.empty_cache()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(1)
seed = 695392 
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', type=str)
parser.add_argument('--dataset', type=str, default=None) # Subject + Timestamp default (see below)
parser.add_argument('--split', type=str, default="1")
parser.add_argument('--action', type=str, choices=['train', 'eval', 'inference', 'CV', "feature_ablation"], default='train')
parser.add_argument('--result_dir', type=str, default="result")
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to model for eval/inference')
parser.add_argument('--eval_epoch', type=int, default=None)

args = parser.parse_args()
all_params = json.load(open(args.config))

# Merge argparse arguments into all_params
for k, v in vars(args).items():
    if v is not None:
        all_params[k] = v
print(all_params)
print()



if all_params["action"] == 'train':
    print(f"Training model:")
    train_nc_paths = all_params["train_nc_paths"]
    test_nc_paths = all_params["test_nc_paths"]
    print(f"  Train sessions: {[p for p in train_nc_paths]}")
    print(f"  Test sessions: {[p for p in test_nc_paths]}")

elif all_params["action"] in ['CV', 'feature_ablation']:
    print()
    print("Running cross-validation, split {}:".format(args.split))
    
    train_nc_paths = all_params[f'split_{args.split}']["train_nc_paths"]
    test_nc_paths = all_params[f'split_{args.split}']["test_nc_paths"]
    
    print(f"  Train sessions: {[p for p in train_nc_paths]}")
    print(f"  Test sessions: {[p for p in test_nc_paths]}")

elif all_params["action"] == 'eval':
    
    if "test_nc_paths" in all_params:
        test_nc_paths = all_params["test_nc_paths"]
    elif "test_nc_paths" in all_params.get(f'split_{args.split}', {}):
        test_nc_paths = all_params[f'split_{args.split}']["test_nc_paths"]
    
    print(f"  Eval sessions: {[p for p in test_nc_paths]}")


elif all_params["action"] == 'inference':
    assert(all_params["model_path"] is not None)
    print(f"Inferring model: {all_params['model_path']}")
    
    if "test_nc_paths" in all_params:
        test_nc_paths = all_params["test_nc_paths"]
    elif "test_nc_paths" in all_params.get(f'split_{args.split}', {}):
        test_nc_paths = all_params[f'split_{args.split}']["test_nc_paths"]
    else:
        raise ValueError("No test_nc_paths found in config for inference.")
    
    # test_nc_paths = all_params[f'split_{args.split}']["test_nc_paths"] # inference on a split
    print(f"  Inference sessions: {[p for p in test_nc_paths]}")


    
    


# Params
lr = all_params.get("learning_rate")
num_epochs = all_params.get("num_epochs")
bz = all_params.get("batch_size")
num_layers = all_params.get("num_layers")
num_f_maps = all_params.get("num_f_maps")
channel_mask_rate = all_params.get("channel_mask_rate")
f1_thresholds = all_params.get("f1_thresholds")
sample_rate = all_params.get("sample_rate", 1)
boundary_weight_schedule = all_params.get("boundary_weight_schedule")
boundary_radius = all_params.get("boundary_radius")



# global
project_root = get_project_root()
mapping_path = project_root / "configs" / "mapping.txt"


# data/
if args.dataset is None:
    args.dataset = os.path.basename(args.config).replace('.json', '')

dataset_dir = project_root / "data" / args.dataset
all_params["dataset_dir"] = str(dataset_dir)

vid_list_file = os.path.join(dataset_dir, f"splits/train.split{args.split}.bundle")
vid_list_file_tst = os.path.join(dataset_dir, f"splits/test.split{args.split}.bundle")

# # CHANGE BACK 
# features_path = os.path.join(dataset_dir, "features/", f"split_{args.split}/")
features_path = os.path.join(dataset_dir, "features/")

gt_path = os.path.join(dataset_dir, "groundTruth/")
trial_mapping_path = os.path.join(dataset_dir, "trial_mapping.json")


# result/
results_dir = project_root / "result" / args.dataset / f"split_{args.split}"
all_params["result_dir"] = str(results_dir)


for d in [features_path, gt_path, Path(vid_list_file).parent]:
    os.makedirs(d, exist_ok=True)




class_to_idx, idx_to_class = load_mapping(mapping_path)
num_classes = len(class_to_idx)
print("num_classes:"+str(num_classes))



# Only relevant for training/cross-validation
if all_params["action"] in ["CV", "train", "feature_ablation"] and not all_params.get("trainDataReady"):  # Skip reloading train data if already converted o
    train_trial_dict = get_trial_dict(all_params, train_nc_paths)

    _ = get_data_dict(
        all_params=all_params,
        nc_paths=train_nc_paths,
        trial_dict=train_trial_dict,
        features_path=features_path,
        gt_path=gt_path,
        idx_to_class=idx_to_class
    )
    write_bundle_list(train_trial_dict, vid_list_file)
else:
    train_trial_dict = dict()



# Relevant for training, eval and inference
test_trial_dict = get_trial_dict(all_params, test_nc_paths)
features_dim = get_data_dict(
    all_params=all_params,
    nc_paths=test_nc_paths,
    trial_dict=test_trial_dict,
    features_path=features_path,
    gt_path=gt_path,
    idx_to_class=idx_to_class
)
write_bundle_list(test_trial_dict, vid_list_file_tst)

# feature_dim determined automatically
print(f"Feature dimension: {features_dim}")
all_params["feature_dim"] = features_dim

trial_mapping = train_trial_dict | test_trial_dict

with open(trial_mapping_path, 'w') as f:
    json.dump(trial_mapping, f, indent=4)
    

if not all_params["action"] == "inference":
    os.makedirs(results_dir, exist_ok=True)
    config_path = os.path.join(results_dir, os.path.basename(args.config))
    with open(config_path, 'w') as outfile:
        json.dump(all_params, outfile, ensure_ascii=False, indent=2)




trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate, f1_thresholds, boundary_weight_schedule, boundary_radius)
if args.action in ["train", "CV", "feature_ablation"]:
    batch_gen = BatchGenerator(num_classes, class_to_idx, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, class_to_idx, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)

    model_dir = results_dir
    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst, all_params)
    


if args.action == "eval":
    batch_gen_tst = BatchGenerator(num_classes, class_to_idx, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)
    
    epoch = all_params["eval_epoch"]
    
    model_path = os.path.join(results_dir, f"epoch-{epoch}.model")
    trainer.model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Loaded model from {model_path} for evaluation at epoch {epoch}")    
    trainer.test(batch_gen_tst, epoch, all_params)



    # trainer.inference(model_path, features_path, batch_gen_tst, num_epochs, trial_mapping, sample_rate, all_params)
    
    

if args.action == "inference":
    batch_gen_tst = BatchGenerator(num_classes, class_to_idx, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst, shuffle=False)
    

    trainer.inference_stepwise(all_params["model_path"], features_path, batch_gen_tst, num_epochs, trial_mapping, sample_rate, all_params)







# if __name__ == '__main__':
#     """
#     TODO: If inference mode, just save 'predictions' to .nc file if labels already not zero. Allow user to override existing labels manually.


#     python moveseg/model/main2.py --config configs/Freddy_CV_fold1.json --method train
#     python moveseg/model/main2.py --config configs/Freddy_CV_fold2.json --method train
#     python moveseg/model/main2.py --config configs/Freddy_CV_fold3.json --method train



#     python moveseg/model/main2.py --config configs/Freddy_CV_fold1.json --method eval --model_path result\Freddy_CV_fold1\epoch-300.model
#     python moveseg/model/main2.py --config configs/Freddy_CV_fold2.json --method eval --model_path result\Freddy_CV_fold2\epoch-300.model
#     python moveseg/model/main2.py --config configs/Freddy_CV_fold3.json --method eval --model_path result\Freddy_CV_fold3\epoch-300.model

#     NOTE: Method has to be specified by the user.
#     - method: train, eval, inference
#         - train: for training with optional eval on test set after every log_freq epochs
#         - eval: for evaluation of a trained model on test set
#         - inference: for inference of a trained model on test set without evaluation (no ground truth needed)




#     NOTE: Diffact has two modes. The first can be controlled by user but indirectly. E.g. set temporal_aug to True/False.
#     Recommendation: Just leave the default settings in config file. The second mode is set internally based on method (no user input).

#     - mode: encoder, decoder-noagg, decoder-agg (can be specified by user for eval/inference, e.g. --mode decoder-agg)
#         - encoder: use encoder predictions only
#         - decoder-noagg: use decoder predictions without temporal augmentation
#         - decoder-agg: use decoder predictions with temporal augmentation (averaging)

#     - mode: train, test
#         - train: for training data (with ground truth)
#         - test: for test/inference data (with or without ground truth)

#     """




