'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-09-17 21:22:20
Email: haimingzhang@link.cuhk.edu.cn
Description: The 3DMM face renderer.
'''

import numpy as np

from .bfm import ParametricFaceModel
from .nvdiffrast import MeshRenderer


class Face3DMMRenderer(object):
    def __init__(self):
        self.face_model = ParametricFaceModel(bfm_folder='./BFM')
        
        fov = 2 * np.arctan(112.0 / 1015.0) * 180 / np.pi
        self.renderer = MeshRenderer(
            rasterize_fov=fov, znear=5.0, zfar=15.0, rasterize_size=int(2 * 112.0)
        )
    
    def forward(self, coeff):
        self.face_model.to(coeff.device)

        pred_vertex, pred_tex, pred_color, pred_lm = \
            self.face_model.compute_for_render(coeff[:, :257])

        pred_mask, _, pred_face = self.renderer(
            pred_vertex, self.face_model.face_buf, feat=pred_color)
        
        return pred_face, pred_mask, pred_lm