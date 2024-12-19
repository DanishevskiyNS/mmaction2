import numpy as np

import torch
from torchvision.transforms import RandomRotation
from torchvision.transforms import functional as F

from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS

@TRANSFORMS.register_module()
class RandomRotateClip(BaseTransform, RandomRotation):
    """
    Класс на основе RandomRotation из Torchvsion для рандомного поворота всего клипа

    Пример использования:
    pipeline = [
        dict(type='UniformSampleFrames', clip_len=48),
        dict(type='PoseDecode'),
        dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
        dict(type='Resize', scale=(-1, 64)),
        dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
        dict(type='Resize', scale=(128, 128), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
        dict(
            type='GeneratePoseTarget',
            sigma=0.6,
            use_score=True,
            with_kp=False,
            with_limb=True,
            skeletons=skeletons),
        dict(type="RandomRotateClip", degrees=20),
        dict(type='FormatShape', input_format='NCTHW_Heatmap'),
        dict(type='PackActionInputs')
    ]

    """
    def __init__(self, degrees, interpolation=F.InterpolationMode.NEAREST, expand=False, center=None, fill=0):
        super().__init__(degrees, interpolation, expand, center, fill)
            
    def forward(self, img, angles):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        channels, _, _ = F.get_dimensions(img)
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        return F.rotate(img, angles, self.interpolation, self.expand, self.center, fill)

    def transform(self, results):
        assert "imgs" in results
        rotation_params = RandomRotateClip.get_params(self.degrees)

        for i in range(len(results["imgs"])):
            rotated_img = self.forward(torch.from_numpy(results["imgs"][i]), rotation_params)
            results["imgs"][i] = rotated_img.numpy()
        return results