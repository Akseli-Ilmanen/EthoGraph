import argparse
import os
import pickle
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Union
import platform

import numpy as np
from omegaconf.dictconfig import DictConfig
import torch
import torch.nn.functional as F
from omegaconf.listconfig import ListConfig


def make_path(output_root, video_path, output_key, ext):
    # extract file name and change the extention
    fname = f'{Path(video_path).stem}_{output_key}{ext}'
    # construct the paths to save the features
    return os.path.join(output_root, fname)

def form_slices(size: int, stack_size: int, step_size: int) -> list((int, int)):
    '''print(form_slices(100, 15, 15) - example'''
    slices = []
    # calc how many full stacks can be formed out of framepaths
    full_stack_num = (size - stack_size) // step_size + 1
    for i in range(full_stack_num):
        start_idx = i * step_size
        end_idx = start_idx + stack_size
        slices.append((start_idx, end_idx))
    return slices


def sanity_check(args: Union[argparse.Namespace, DictConfig]):
    '''Checks user arguments.

    Args:
        args (Union[argparse.Namespace, DictConfig]): Parsed user arguments
    '''
    if 'device_ids' in args:
        print('WARNING:')
        print('Running feature extraction on multiple devices in a _single_ process is no longer supported.')
        print('To use several GPUs, you simply need to start the extraction with another GPU ordinal.')
        print('For instance, in one terminal: `device="cuda:0"` and `device="cuda:1"` in the second, etc.')
        print(f'Your device specification (device_ids={args.device_ids}) is converted to `device="cuda:0"`.')
        args.device = 'cuda:0'
    if 'cuda' in args.device and not torch.cuda.is_available():
        print(f'A GPU was attempted to use but the system does not have one. Going to use CPU...')
        args.device = 'cpu'
    assert args.file_with_video_paths or args.video_paths, '`video_paths` or `file_with_video_paths` must be specified'
    filenames = [Path(p).stem for p in form_list_from_user_input(args.video_paths, args.file_with_video_paths)]
    assert len(filenames) == len(set(filenames)), 'Non-unique filenames. See video_features/issues/54'
    
    if 'extraction_fps' in args and 'extraction_total' in args:
        assert not (args.extraction_fps is not None and args.extraction_total is not None),\
            '`fps` and `total` is mutually exclusive'

    # patch_output_paths
    # preprocess paths
    subs = [args.feature_type]
    if hasattr(args, 'model_name'):
        subs.append(args.model_name)
        # may add `finetuned_on` item
    real_output_path = args.output_path

    
    for p in subs:
        # some model use `/` e.g. ViT-B/16
        real_output_path = os.path.join(real_output_path, p.replace("/", "_"))

    args.output_path = real_output_path



def form_list_from_user_input(
        video_paths: Union[str, ListConfig, None] = None,
        file_with_video_paths: str = None,
        to_shuffle: bool = True,
    ) -> list:
    '''User specifies either list of videos in the cmd or a path to a file with video paths. This function
       transforms the user input into a list of paths. Files are expected to be formatted with a single
       video-path in each line.

    Args:
        video_paths (Union[str, ListConfig, None], optional): a list of video paths. Defaults to None.
        file_with_video_paths (str, optional): a path to a file with video files for extraction.
                                               Defaults to None.
        to_shuffle (bool, optional): if the list of paths should be shuffled. If True is should prevent
                                     potential worker collisions (two workers process the same video)

    Returns:
        list: list with paths
    '''
    if file_with_video_paths is None:
        path_list = [video_paths] if isinstance(video_paths, str) else list(video_paths)
        # TODO: the following `if` could be redundant
        # ListConfig does not support indexing with tensor scalars, e.g. tensor(1, device='cuda:0')
        if isinstance(video_paths, ListConfig):
            path_list = list(path_list)
    else:
        with open(file_with_video_paths) as rfile:
            # remove carriage return
            path_list = [line.replace('\n', '') for line in rfile.readlines()]
            # remove empty lines
            path_list = [path for path in path_list if len(path) > 0]

    # sanity check: prints paths which do not exist
    for path in path_list:
        if not Path(path).exists():
            print(f'The path does not exist: {path}')

    if to_shuffle:
        random.shuffle(path_list)

    return path_list


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    # Determine the platform on which the program is running
    if platform.system().lower() == 'windows':
        result = subprocess.run(['where', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ffmpeg_path = result.stdout.decode('utf-8').splitlines()[0]
    else:
        result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ffmpeg_path = result.stdout.decode('utf-8').splitlines()[0]
    return ffmpeg_path



def build_cfg_path(feature_type: str) -> os.PathLike:
    '''Makes a path to the default config file for each feature family.

    Args:
        feature_type (str): the type (e.g. 'vggish')

    Returns:
        os.PathLike: the path to the default config for the type
    '''
    path_base = Path('./configs')
    path = path_base / f'{feature_type}.yml'
    return path


def dp_state_to_normal(state_dict):
    '''Converts a torch.DataParallel checkpoint to regular'''
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module'):
            new_state_dict[k.replace('module.', '')] = v
    return new_state_dict


def load_numpy(fpath):
    return np.load(fpath)

def write_numpy(fpath, value):
    return np.save(fpath, value)

def load_pickle(fpath):
    return pickle.load(open(fpath, 'rb'))

def write_pickle(fpath, value):
    return pickle.dump(value, open(fpath, 'wb'))
