'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-09-17 21:58:09
Email: haimingzhang@link.cuhk.edu.cn
Description: The training PL model for the RealFaceGenerator.
'''
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
        self.src_img = None
        self.dst_img = None
        
        self.automatic_optimization = False 
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
        torch.backends.cudnn.benchmark = True

    def forward(self, I_source, I_target,mask):
        img = self.G(I_source, I_target,mask)[0]
        return img
    
    @torch.no_grad()
    def process_cmd(self):
        if self.s2c is None:
            return
        if not self.s2c.empty():
            msg = self.s2c.get()
            if msg == "preview":
                self.send_previw()
            elif msg == "random_z":
                self.src_img = None
                self.dst_img = None
                self.src_latent = None
            elif msg == "stop":
                torch.save(self.G.state_dict(),"./weights/G.pth")
                torch.save(self.D.state_dict(),"./weights/D.pth")
                self.trainer.should_stop = True
            else:
                pass
                     
            
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
        I_source, mask_source, I_target, mask_target = batch

        if self.src_img == None:
            self.src_img = I_source[:3]
            self.dst_img = I_target[:3]
            self.dst_msk = mask_target[:3]
            
        self.process_cmd()

        blur_sigma = max(1 - self.global_step / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        I_swapped_high, I_swapped_low = self.G(I_source, I_target, mask_target)

        # adversarial
        fake_output = self.run_D(I_swapped_high, blur_sigma = blur_sigma)
        real_output = self.run_D(I_target, blur_sigma = blur_sigma)

        G_dict = {
            "I_source": I_source,
            "I_target": I_target,
            "I_swapped_high": I_swapped_high,
            "I_swapped_low": I_swapped_low,
            "mask_target": mask_target,
            "d_fake": fake_output,
            "d_real": real_output
        }
        
        g_loss = self.loss.get_loss_G(G_dict)
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()

        ###########
        # train D #
        ###########
        I_target.requires_grad_()
        d_true = self.run_D(I_target,blur_sigma = blur_sigma)
        d_fake = self.run_D(I_swapped_high.detach(),blur_sigma = blur_sigma)

        D_dict = {
            "d_true": d_true,
            "d_fake": d_fake,
            "I_target": I_target
        }

        d_loss = self.loss.get_loss_D(D_dict)
        
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()
        self.log_dict(self.loss.loss_dict)

    def configure_optimizers(self):
        optimizer_list = []
        optim_params = self.config.optimizer.params

        optimizer_g = torch.optim.AdamW(self.G.parameters(), **optim_params)
        optimizer_list.append({"optimizer": optimizer_g})
        optimizer_d = torch.optim.AdamW(self.D.parameters(), lr=optim_params.lr * 0.8)
        optimizer_list.append({"optimizer": optimizer_d})
        
        return optimizer_list
