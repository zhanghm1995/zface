'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-09-17 21:58:09
Email: haimingzhang@link.cuhk.edu.cn
Description: The training PL model for the RealFaceGenerator.
'''

import os
import os.path as osp
from imageio import save
import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader
import platform
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from torch_utils.ops import upfirdn2d
from models.real_face_generator import RealFaceGenerator
from models.discriminator import ProjectedDiscriminator
from torch import nn
from dataset import *
from losses.real_face_loss import RealFaceLoss
from models.gradnorm import normalize_gradient
from utils.deep3dface.face_renderer import Face3DMMRenderer


mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


class RealFaceModel(pl.LightningModule):
    def __init__(self, cfg=None):
        super(RealFaceModel, self).__init__()
        self.config = cfg
        
        self.size = cfg["image_size"]
        self.z_dim = cfg["latent_dim"]

        self.G = RealFaceGenerator(activation=cfg["activation"])
        self.D = ProjectedDiscriminator(im_res=self.size, backbones=['deit_base_distilled_patch16_224',
                                                                     'tf_efficientnet_lite0'])    
        self.blur_init_sigma = 2
        self.blur_fade_kimg = 200

        self.loss = RealFaceLoss(cfg)
        self.generated_img = None
        
        self.automatic_optimization = False 
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
        torch.backends.cudnn.benchmark = True

        self.face3dmm_renderer = Face3DMMRenderer()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, I_source, I_target,mask):
        img = self.G(I_source, I_target,mask)[0]
        return img
    
    @torch.no_grad()
    def send_previw(self):
        output = self.G.inference(self.src_img, self.dst_img,self.dst_msk)
        result =  []
        for src, dst,dst_msk, out  in zip(self.src_img.cpu(),self.dst_img.cpu(),self.dst_msk.cpu(),output.cpu()):
            src = unnormalize(src)
            dst = unnormalize(dst)
            out = unnormalize(out)
            dst_msk = torch.ones_like(dst) * dst_msk
            result = result + [src, dst, dst_msk, out]
        self.c2s.put({'op':"show",'previews': result})
     
    def run_D(self,img,blur_sigma = 0):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
            img = upfirdn2d.filter2d(img, f / f.sum())
        return self.D(img)
            
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers(use_pl_optimizer=True)
        I_source, I_target, mask_target, I_gt = self._prepare_data(batch)

        # I_swapped_high, I_swapped_low = self.G(I_source, I_target, mask_target)

        # # adversarial
        # fake_output = self.run_D(I_swapped_high)
        # real_output = self.run_D(I_gt)

        # G_dict = {
        #     "I_gt": I_gt,
        #     "I_swapped_high": I_swapped_high,
        #     "I_swapped_low": I_swapped_low,
        #     "mask_target": mask_target,
        #     "d_fake": fake_output,
        #     "d_real": real_output
        # }
        
        # g_loss = self.loss.get_loss_G(G_dict)
        # opt_g.zero_grad(set_to_none=True)
        # self.manual_backward(g_loss)
        # opt_g.step()

        # ###########
        # # train D #
        # ###########
        # I_gt.requires_grad_()
        # d_true = self.run_D(I_gt)
        # d_fake = self.run_D(I_swapped_high.detach())

        # D_dict = {
        #     "d_true": d_true,
        #     "d_fake": d_fake,
        #     "I_gt": I_gt
        # }

        # d_loss = self.loss.get_loss_D(D_dict)
        
        # opt_d.zero_grad(set_to_none=True)
        # self.manual_backward(d_loss)
        # opt_d.step()
        # self.log_dict(self.loss.loss_dict)

        if batch_idx % 500 == 0:
            self.save_images(I_source, I_target, I_gt)
    
    def save_images(self, I_source, I_target, I_gt):
        save_dir = osp.join(self.logger.log_dir, "vis", f"epoch_{self.current_epoch:04d}")
        save_path = osp.join(save_dir, f"iteration_{self.global_step:08d}.jpg")
        
        os.makedirs(save_dir, exist_ok=True)
        vis_images = torch.stack([I_source, I_target, I_gt], dim=0) # (N, B, 3, H, W)
        vis_images = vis_images[:, :10, ...]
        vis_images = vis_images.reshape(-1, *vis_images.shape[2:])

        vis_images = unnormalize(vis_images)
        
        torchvision.utils.save_image(vis_images, save_path, nrow=10, padding=0, normalize=False)


    def _prepare_data(self, batch):
        for key, value in batch.items():
            print(key, value.shape, value.dtype)

        target_semantics = batch['target_semantics']
        ## Rendering the target semantics
        target_rendered_face, target_mask, _ = self.face3dmm_renderer.forward(target_semantics.to(torch.float32))
        
        gt_full_image = batch['gt_image'] # (B, 3, 224, 224)
        gt_image = gt_full_image * target_mask
        
        # resize to (256, 256)
        gt_image = F.interpolate(gt_image, size=256, mode='bilinear')

        target_image = F.interpolate(target_rendered_face, size=256, mode='bilinear')
        target_image = self.normalize(target_image)

        source_image = F.interpolate(batch['reference_image'], size=256, mode='bilinear')

        return source_image, target_image, target_mask, gt_image

    def configure_optimizers(self):
        optimizer_list = []
        optim_params = self.config.optimizer.params

        optimizer_g = torch.optim.AdamW(self.G.parameters(), **optim_params)
        optimizer_list.append({"optimizer": optimizer_g})
        optimizer_d = torch.optim.AdamW(self.D.parameters(), lr=optim_params.lr * 0.8)
        optimizer_list.append({"optimizer": optimizer_d})
        
        return optimizer_list
