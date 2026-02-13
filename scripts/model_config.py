import os
import json
import copy
import subprocess
import sys
from datetime import datetime
import psutil
from pathlib import Path
import traceback
import importlib
import sys
from ethograph.model.dataset import save_config
from ethograph import get_project_root


params_rigid = {
"Note": "Purge, stich and other changepoint_params determiend in configs/changepoints_settings.yaml",
"good_s3d_feats": None,
"changepoint_feats": {
   "sigmas": [2.0, 3.0, 5.0],
   "merge_changepoints": True,
},
"root_data_dir": "./data",
"split_id": 1,
"sample_rate": 1,
"num_layers": 10,
"num_f_maps": 64,
"r1": 2,
"r2": 2,
"channel_mask_rate": 0.3,
"batch_size":1,
"learning_rate":0.0005,
"num_epochs":100,
"eval_epoch": 100,
"log_freq":10, # At how many epochs, the model is saved
"f1_thresholds": [0.5, 0.75, 0.9],  # IoU thresholds for F1 score calculation
"boundary_radius": 2, # Window = 2*radius+1
"boundary_weight_schedule": {
            0: 0.0,    # Let encoder learn first
            10: 0.5,  
            20: 1.0,   
            30: 1.5,  
            40: 2.0  
        }
}


if __name__ == "__main__":



   # need to comment out for train-all
   action="inference" # "train", "inference", "CV", "ablation"
   # eval run manually via terminal
   
   trainDataReady = False
   
   # model_path = r"D:\Akseli\Code\ethograph\configs\model\Freddy_train_20251021_164220.json" # only for inference mode
   # model_path = os.path.join(get_project_root(), "configs", "model", "Ivy_train_20260202_191138_epoch-100.model")


   target_individual = "Crow1" # predict labels for this individual
   
   cp_kwargs = {
      "individuals": target_individual,
      "keypoints": "beakTip",
   }
   feat_kwargs = {
      "keypoints": ["beakTip", "stickTip"],
      "individuals": target_individual,
   }
   
   

   mapping_file = os.path.join(get_project_root(), "configs", "mapping.txt")


   nc_paths = [      
      r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250527_01\behav\Trial_data.nc", 
      r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250527_02\behav\Trial_data.nc", 
      r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250528_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250526_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250526_02\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250528_02\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250529_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250530_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250602_01\behav\Trial_data.nc"
      


      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250306_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250309_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250503_02\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250514_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250504_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250505_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250307_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250308_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250506_02\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250507_02\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250507_03\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250508_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250508_02\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250509_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250512_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250513_01\behav\Trial_data.nc",      
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250515_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250516_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250519_01\behav\Trial_data.nc",
      
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250521_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250522_01\behav\Trial_data.nc"             
   ]
         
   
   params_dynamic = copy.deepcopy(params_rigid)
   params_dynamic['action'] = action
   params_dynamic['mapping_file'] = mapping_file
   params_dynamic['target_individual'] = target_individual
   params_dynamic['cp_kwargs'] = cp_kwargs
   params_dynamic['feat_kwargs'] = feat_kwargs
   params_dynamic["trainDataReady"] = trainDataReady
         
         
   # # ---------- Train on all data/Inference with trainAll -----------
   if action in ["train", "inference"]:
      if action == "train":
         params_dynamic['train_nc_paths'] = nc_paths
         params_dynamic['test_nc_paths'] = [nc_paths[0]]  # For compatibility, no eval
      if action in ["inference"]:
         params_dynamic['test_nc_paths'] = nc_paths  # Inference on all sessions
      config_path = save_config(params_dynamic, 'configs/model', action)
      
      if action == "train":
         print("Next run: \npython scripts/model_run.py --config {} --action train".format(config_path))
      elif action == "inference":
         print("Next run: \npython scripts/model_run.py --config {} --action inference --model_path {}".format(config_path, str(model_path)))

   if action == "CV":
      env = os.environ.copy()

      num_sessions = len(nc_paths)


      for fold_id in range(num_sessions):
         
         # For each fold, use one session for testing and the rest for training
         test_nc_paths = [nc_paths[fold_id]]
         train_nc_paths = [nc_paths[i] for i in range(num_sessions) if i != fold_id]
         params_dynamic[f'split_{fold_id+1}'] = {"train_nc_paths": train_nc_paths, "test_nc_paths": test_nc_paths}


      config_path = save_config(params_dynamic, 'configs/model', action)
         
      get_project_root()
   
      for fold_id in range(num_sessions):      
          result = subprocess.run(
            [sys.executable, str(get_project_root()  / 'scripts' / 'model_run.py'), '--action', 'CV', '--config', config_path, '--split', str(fold_id+1)],
            env=env,
            text=True
          )
         
   
   # if action == "ablation":  
   
   #    env = os.environ.copy()

   #    # later manually add 1 condition for all s3d
   #    # conditions = ["no_s3d", "no_changepoint", "no_kinematic", "full"]
   #    conditions = ["no_circle_loss", "no_boundary_weighting"]
      
      
      
   #    fold_id = 0
   #    test_nc_paths = [nc_paths[fold_id]]
   #    train_nc_paths = [nc_paths[i] for i in range(len(nc_paths)) if i != fold_id]      

      
   #    for i, cond in enumerate(conditions):
   #       params_dynamic[f'split_{i+1}'] = {"feature_ablation_condition": cond,
   #                                         "train_nc_paths": train_nc_paths, "test_nc_paths": test_nc_paths}
         
   #       if cond == "no_circle_loss":
   #          params_dynamic["circle_loss"] = False
   #       elif cond == "no_boundary_weighting":
   #          params_dynamic["boundary_weight_schedule"] = {
   #             0: 0.0
   #          }
         
         



   #    config_path = save_config(params_dynamic, 'configs', action)
         
         
   
   #    for i in range(len(conditions)):      
   #       result = subprocess.run(
   #          [sys.executable, str(get_project_root() / 'ethograph' / 'model' / 'model_run'), '--action', 'CV', '--config', config_path, '--split', str(i+1)],
   #          env=env,
   #          text=True
   #       ) 
            
