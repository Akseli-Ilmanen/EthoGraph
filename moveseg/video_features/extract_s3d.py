from typing import Dict

import numpy as np
import torch
import torchvision
from moveseg.video_features.base_extractor import BaseExtractor
from moveseg.video_features.s3d import S3D
from moveseg.video_features.transforms import CenterCrop, Resize, ToFloatTensorInZeroOne
from torchvision.io.video import read_video
from moveseg.video_features.utils import form_slices


class ExtractS3D(BaseExtractor):

    def __init__(self, args) -> None:
        # init the BaseExtractor
        super().__init__(
            feature_type=args.feature_type,
            output_path=args.output_path,
            device=args.device,
        )
        # (Re-)Define arguments for this class
        self.stack_size = 64 if args.stack_size is None else args.stack_size
        

        # Step size: The number of frames to step before extracting the next features
        # Given that we want features per frame, we set these to 1
        self.step_size = 1
        
        
        # normalization is not used as per: https://github.com/kylemin/S3D/issues/4
        self.transforms = torchvision.transforms.Compose([
            ToFloatTensorInZeroOne(),
            Resize(224),
            CenterCrop((224, 224))
        ])

        self.output_feat_keys = [self.feature_type]
        self.name2module = self.load_model()

    @torch.no_grad()
    def extract(self, video_path: str) -> Dict[str, np.ndarray]:
        """Extracts features for a given video path.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: feature name (e.g. 'fps' or feature_type) to the feature tensor
        """
    

        # read a video
        rgb, audio, info = read_video(video_path, pts_unit='sec')
        
        # add black frames to the end of the video
        # See https://github.com/v-iashin/video_features/issues/149

        assert self.stack_size % 2 == 1, "stack_size must be odd"
        num_black_frames_pre = self.stack_size // 2
        num_black_frames_post = self.stack_size // 2
        black_frames_pre = torch.zeros(num_black_frames_pre, rgb.size(1), rgb.size(2), rgb.size(3), dtype=rgb.dtype)
        black_frames_post = torch.zeros(num_black_frames_post, rgb.size(1), rgb.size(2), rgb.size(3), dtype=rgb.dtype)
        rgb = torch.cat([black_frames_pre, rgb, black_frames_post], dim=0)




        # prepare data (first -- transform, then -- unsqueeze)
        rgb = self.transforms(rgb)  # could run out of memory here
        rgb = rgb.unsqueeze(0)
        # slice the stack of frames
        slices = form_slices(rgb.size(2), self.stack_size, self.step_size)

        vid_feats = []

        for stack_idx, (start_idx, end_idx) in enumerate(slices):
            # inference
            rgb_stack = rgb[:, :, start_idx:end_idx, :, :].to(self.device)
            output = self.name2module['model'](rgb_stack, features=True)
            vid_feats.extend(output.tolist())


        feats_dict = {
            self.feature_type: np.array(vid_feats),
        }

        return feats_dict

    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints, sends them to the device.

        Raises:
            NotImplementedError: if a model is not implemented.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        s3d_kinetics400_weights_torch_path = './moveseg/video_features/checkpoint/S3D_kinetics400_torchified.pt'
        model = S3D(num_class=400, ckpt_path=s3d_kinetics400_weights_torch_path)
        model = model.to(self.device)
        model.eval()

        return {
            'model': model,
        }

