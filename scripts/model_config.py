import os
import json
import copy
import subprocess
import sys
from datetime import datetime
import psutil
import traceback
import importlib
import sys
from ethograph.model.dataset import save_config
from ethograph.utils.paths import get_project_root


params_rigid = {
"good_s3d_feats": None, # Already exclude in file generation
"min_motif_len": 10, # Same as purge value, but also applied after changepoint correction, for toss, I in changepoint set this to 5
"stitch_gap_len": 3, # 000222000222333000 -> 0002222222333000
"changepoint_params": {
   "sigmas": [2.0, 3.0, 5.0],
   "merge_changepoints": True,
   "max_expansion": 10.0, # in samples
   "max_shrink": 10.0, # in samples      
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
"num_epochs":1,
"eval_epoch": 1,
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
   
   # model_path = r"D:\Akseli\Code\ethograph\result\Freddy_train_20251021_164220\split_1\epoch-100.model" # only for inference mode
   model_path = r"D:\Akseli\Code\ethograph\configs\model\Ivy_train_20260128_171450_epoch-100.model"
   
   
   


   target_individual = "Ivy" # predict labels for this individual
   
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
      # r"C:\Users\FM\Desktop\trainFreddy\Trial_data2601.nc",
      # r"C:\Users\FM\Desktop\trainFreddy\Trial_data2701.nc",
      # r"C:\Users\FM\Desktop\trainFreddy\Trial_data2702.nc",
      # r"C:\Users\FM\Desktop\trainFreddy\Trial_data2801.nc",
      # r"C:\Users\Julius\Desktop\FreddyTrain\small1.nc",
      # r"C:\Users\Julius\Desktop\FreddyTrain\small2.nc",
      # r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250527_01\behav\Trial_data.nc", 
      # r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250527_02\behav\Trial_data.nc", 
      # r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250528_01\behav\Trial_data.nc",
      # r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250526_01\behav\Trial_data.nc",
      # r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250526_02\behav\Trial_data.nc",
      # r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250528_02\behav\Trial_data.nc",
      # r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250529_01\behav\Trial_data.nc",
      # r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250530_01\behav\Trial_data.nc",
      # r"D:\Alice\AK_data\derivatives\sub-03_id-Freddy\ses-000_date-20250602_01\behav\Trial_data.nc",
      
      # r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250306_01\behav\Trial_data.nc",
      # r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250309_01\behav\Trial_data.nc",
      # r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250503_02\behav\Trial_data.nc",
      # r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250514_01\behav\Trial_data.nc",
      r"D:\Alice\AK_data\derivatives\sub-01_id-Ivy\ses-000_date-20250504_01\behav\Trial_data.nc"
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
         print("Next run: \npython scripts/model_run.py --config {} --action inference --model_path {}".format(config_path, model_path))

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
            [sys.executable, str(get_project_root() / 'ethograph' / 'model' / 'model_run'), '--action', 'CV', '--config', config_path, '--split', str(fold_id+1)],
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
            
