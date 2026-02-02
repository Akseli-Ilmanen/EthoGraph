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
from ethograph import TrialTree
from ethograph.features.preprocessing import clip_by_percentiles, z_normalize, interpolate_nans
from ethograph.features.changepoints import more_changepoint_features, merge_changepoints
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


def extract_features_per_trial(ds, all_params):
    """
    Extracts and concatenates changepoint and feature data for a single trial from a dataset.
    This function selects the data corresponding to the specified trial, removes any padding.
    ----------
    ds : xarray.Dataset (single trial)
    Returns
    -------
    tuple of np.ndarray
        changepoint_feats: 2D array of shape (time, num_changepoint_features)
        features: 2D array of shape (time, num_features)
    """
    
    changepoint_sigmas = all_params["changepoint_params"]["sigmas"]
    feat_kwargs = all_params["feat_kwargs"]
    cp_kwargs = all_params["cp_kwargs"]
    good_s3d_feats = all_params["good_s3d_feats"]
    
    

    if all_params["changepoint_params"]["merge_changepoints"]:
        ds, target_feature = merge_changepoints(ds)
    

    cp_ds = ds.sel(**cp_kwargs).filter_by_attrs(type="changepoints")

    
    

    cp_list = []
    for var in cp_ds.data_vars:
        
        if not all_params["changepoint_params"]["merge_changepoints"]:
            target_feature = cp_ds[var].attrs["target_feature"]
            
        targ_feat_vals = ds[target_feature].sel(**cp_kwargs).values
        cp_data = cp_ds[var].squeeze().values
        
        output = more_changepoint_features(cp_data, sigmas=changepoint_sigmas, targ_feat_vals=targ_feat_vals)
        cp_list.append(output)
    cp_feats = np.hstack(cp_list)

    if good_s3d_feats is None: # or all_params["split_5"]["feature_ablation_condition"] == "all_s3d":
        s3d = ds.s3d.values
    else:
        s3d = ds.s3d.sel(s3d_dims=good_s3d_feats).values
    
    
    # s3d = ds.s3d.values
    ds = ds.drop_vars("s3d")
    
    feat_ds = ds.sel(**feat_kwargs).squeeze().filter_by_attrs(type="features")
    features = feat_ds.to_stacked_array('features', sample_dims=['time']).values # flatten across non-time dimensions
    shape1 = features.shape
    features = features[:, ~np.all(np.isnan(features), axis=0)]
    shape2 = features.shape
    
    # if shape1[1] != shape2[1]:
    #     print(f"\nWarning: Dropped {shape1[1]-shape2[1]} all-NaN feature columns.")

    return cp_feats, features, s3d


# TO DO, figure out smart way to specify individual in feat_kwargs, else. 
def get_feature_names(ds, all_params):
    changepoint_sigmas = all_params["changepoint_params"]["sigmas"]
    changepoint_names = []

    for var in ds.filter_by_attrs(type="changepoints").data_vars:
        changepoint_names.extend([f"{var}_binary"])
        changepoint_names.extend([f"{var}_σ={sigma}" for sigma in changepoint_sigmas])
        changepoint_names.extend([f"{var}_segIDs"])

    feat_kwargs = all_params["feat_kwargs"]
    feat_da = ds.sel(**feat_kwargs).squeeze().filter_by_attrs(type="features")

    
    feature_var_names = []
    for var_name in feat_da.data_vars:
        var_data = feat_da[var_name]
        non_time_dims = [dim for dim in var_data.dims if dim != 'time' and dim != 'trials']
        if len(non_time_dims) == 0:
            feature_var_names.append(var_name)
        elif len(non_time_dims) == 1:
            dim_name = non_time_dims[0]
            dim_coords = var_data[dim_name].values
            for coord in dim_coords:
                feature_var_names.append(f"{var_name}_{coord}")
        else:
            dim_coords_lists = [var_data[dim].values for dim in non_time_dims]
            for coord_combo in itertools.product(*dim_coords_lists):
                name_parts = [var_name] + [str(c) for c in coord_combo]
                feature_var_names.append("_".join(name_parts))
    return changepoint_names + feature_var_names


def get_data_dict(all_params, nc_paths, trial_dict, features_path=None, gt_path=None, idx_to_class=None):
    


    feature_dim = None
        
    print(f'Loading Dataset ...')
    for hash_key in trial_dict.keys():
        nc_path = trial_dict[hash_key]['nc_path']
        print(f"Processing {nc_path}, hash key: {hash_key}")
        dt = TrialTree.open(nc_path)

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
        dt = TrialTree.open(nc_path)
        
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
