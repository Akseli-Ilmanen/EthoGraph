"""Modified from DiffAct so that features can be directly passed from .nc file"""
import os
from xml.sax.handler import all_features
import torch
import random
import glob
import hashlib
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm
from ethograph.utils.io import extract_features_per_trial, TrialTree, extract_variable_flat
from ethograph.features.preprocessing import clip_by_percentiles, z_normalize, interpolate_nans
import random
import argparse
import shutil
from datetime import datetime
import json

def save_config(all_params, folder='configs', action="train"):
    if not os.path.exists(folder):
        os.makedirs(folder)   

    ID = all_params["target_individual"]

    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(folder, f'{ID}_{action}_{time}.json')
    print(f"Config saved: {config_path}")
    with open(config_path, 'w') as outfile:
        json.dump(all_params, outfile, ensure_ascii=False, indent=2)
        
        
    return config_path




def get_file_hash(filepath, hash_length=8):
    """Generate a short hash from file path for use as dictionary key"""
    # Deterministic -> same path gives same hash
    return hashlib.md5(str(Path(filepath).resolve()).encode()).hexdigest()[:hash_length]


def write_bundle_list(trial_dict, bundle_path):
    bundle_list = [f"{key}_{trial}" for key, val in trial_dict.items() for trial in val["trials"]]
    
    if os.path.exists(bundle_path):
        os.remove(bundle_path)
    
    with open(bundle_path, "w") as f:
        for item in bundle_list:
            f.write(f"{item}.txt\n")

def get_data_dict(all_params, nc_paths, trial_dict, features_path=None, gt_path=None, idx_to_class=None):
    


    feature_dim = None
        
    print(f'Loading Dataset ...')
    for hash_key in trial_dict.keys():
        nc_path = trial_dict[hash_key]['nc_path']
        print(f"Processing {nc_path}, hash key: {hash_key}")
        dt = TrialTree.load(nc_path)

        for trial_num in tqdm(trial_dict[hash_key]['trials']):
            

            ds = dt.trial(trial_num)

            # In inference (data is unlabelled), labels should be all zeros
            individual = all_params["target_individual"]
            labels = np.array(ds.sel(individuals=individual).labels.values)


            # B - Batch, T - Time, F - Feature
            try:
                changepoint_features, features, s3d = extract_features_per_trial(ds, all_params) # (T, F)
            except Exception as e:
                print(f"  ERROR in extract_features_per_trial for session {nc_path}, trial {trial_num}: {e}")
                raise

 
            features = interpolate_nans(features, axis=0) 
            features = clip_by_percentiles(features, percentile_range=(2, 98))
 

            
            
            
            split = all_params["split"]
            
            

            condition = all_params.get(f'split_{split}', {}).get('feature_ablation_condition', 'full')
            if condition not in ("no_changepoint", "no_kinematic", "no_s3d", "full"):
                condition = "full"
            
            if condition == "no_changepoint":
                all_features = np.concatenate([features, s3d], axis=1)
            elif condition == "no_kinematic":
                all_features = np.concatenate([changepoint_features, s3d], axis=1)
            elif condition == "no_s3d":
                all_features = np.concatenate([changepoint_features, features], axis=1)
            elif condition in ["full", "all_s3d"]:
                all_features = np.concatenate([changepoint_features, features, s3d], axis=1)
            
     
            all_features = z_normalize(all_features)
     

            # Capture feature dimension from first trial
            if feature_dim is None:
                feature_dim = all_features.shape[1]  # Number of features (F)
                if condition == "no_changepoint":
                    print(f"\nKinematic features: {features.shape[1]}, S3D features: {s3d.shape[1]}")
                elif condition == "no_kinematic":
                    print(f"\nN changepoint features: {changepoint_features.shape[1]}, S3D features: {s3d.shape[1]}")
                elif condition == "no_s3d":
                    print(f"\nN changepoint features: {changepoint_features.shape[1]}, kinematic features: {features.shape[1]}")
                else:
                    print(f"\nN changepoint features: {changepoint_features.shape[1]}, kinematic features: {features.shape[1]}, S3D features: {s3d.shape[1]}")
                    
                


            features = all_features.T # (F, T)
            np.save(os.path.join(features_path, f'{hash_key}_{trial_num}.npy'), features)

            labels_text = [idx_to_class[int(label_num)] for label_num in labels]
            np.savetxt(os.path.join(gt_path, f'{hash_key}_{trial_num}.txt'), labels_text, fmt='%s')


    return feature_dim



def get_trial_dict(all_params, nc_paths) -> dict:
    trial_dict = {}

    for nc_path in nc_paths:
        hash_key = get_file_hash(nc_path)
        dt = TrialTree.load(nc_path)
        
        valid_trials = []
        for node in dt.children.values():
            trial_num = node.ds.attrs['trial']
            individual = all_params["target_individual"]
            
            if all_params["action"] in ['train', 'eval']:
                labels = node.ds["labels"].sel(individuals=individual).values
                if np.all(labels == 0):
                    continue
                    
            if all_params["action"] == 'inference':
                stick_pos = node.ds.position.sel(
                    keypoints='stickTip', space='x', individuals=individual
                ).values
                valid = stick_pos[~np.isnan(stick_pos) & (stick_pos != 0)]
                if valid.size < 200:
                    continue
            
            valid_trials.append(int(trial_num)) # WILL not work if trial nums are not integers
        
        trial_dict[hash_key] = {
            'nc_path': nc_path,
            'trials': sorted(valid_trials)
        }
    
    return trial_dict



    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
