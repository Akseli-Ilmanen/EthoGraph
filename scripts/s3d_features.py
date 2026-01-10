from omegaconf import OmegaConf

from tqdm import tqdm
import sys
import os
from contextlib import redirect_stdout
from moveseg.video_features.utils import build_cfg_path, form_list_from_user_input, sanity_check
from moveseg.video_features.extract_s3d import ExtractS3D as Extractor



def s3d_features(args_cli):
    args_cli.feature_type = "s3d" # hard-coded
    args_yml = OmegaConf.load(build_cfg_path(args_cli.feature_type))
    args = OmegaConf.merge(args_yml, args_cli)  # the latter arguments are prioritized
    # OmegaConf.set_readonly(args, True)
    sanity_check(args)

    # verbosing with the print -- haha (TODO: logging)
    print(OmegaConf.to_yaml(args))
    print(f'Saving features to {args.output_path}')
    print('Device:', args.device)
    print("stack size: ", args.stack_size)

    extractor = Extractor(args)

    # unifies whatever a user specified as paths into a list of paths
    video_paths = form_list_from_user_input(args.video_paths, args.file_with_video_paths, to_shuffle=True)
    
    

    print(f'The number of specified videos: {len(video_paths)}')

    for video_path in tqdm(video_paths):
        extractor._extract(video_path)  # note the `_` in the method name



if __name__ == '__main__':


    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file_path = os.path.join(project_root, "logging", "s3d_logs.txt")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Redirect stdout to the log file
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        with redirect_stdout(log_file):
            
            args_cli = OmegaConf.from_cli()
            s3d_features(args_cli)


# """
# python moveseg/model/main.py file_with_video_paths=path/to/video_list.txt output_path=path/to/output_dir
# """