import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import DeiTModel
from torch.nn.modules.container import ModuleDict, ParameterDict
from typing import Dict, Union, Optional


def weights_init_kaiming(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class DEIT_Vanilla_Fusion(nn.Module):
    def __init__(self,
                 cfg: Dict[str, Union[int, str]], 
                 fabric: any) -> None:
        super(DEIT_Vanilla_Fusion, self).__init__()
        self.cfg = cfg
        self.fabric = fabric
        hidden_size = self.cfg.vit_embed_dim 
        self.feat_dim = self.cfg.vit_embed_dim # Fuse all 3 modalities into one via averaging 

        self.vanilla_fusion_transformers = nn.TransformerEncoderLayer(d_model=hidden_size,
                                    nhead=self.cfg.model_num_heads).to(self.fabric.device)

        print("Loading pretrained transformer...")

        if self.cfg.pretrained_model == "distilled-384": 
            self.transformer = DeiTModel.from_pretrained(
                'facebook/deit-base-distilled-patch16-384')
        elif self.cfg.pretrained_model == "distilled-224":
            self.transformer = DeiTModel.from_pretrained(
                'facebook/deit-base-distilled-patch16-224')
        elif self.cfg.pretrained_model == "distilled-224-small":
            self.transformer = DeiTModel.from_pretrained(
                'facebook/deit-small-distilled-patch16-224')
        else:
            raise ValueError("Invalid pretrained model")

        self.fabric.to_device(self.transformer)
        self.transformer = self.transformer.eval()

        for param in self.transformer.parameters():
            param.requires_grad = False
        print("Transformer loaded")

        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad = False
        self.bottleneck.apply(weights_init_kaiming)

        if len(self.cfg.gpus) > 1:
            self.bottleneck = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.bottleneck, process_group=dist.group.WORLD)

        self.decoder = nn.Linear(
            self.feat_dim,
            self.cfg.model_decoder_output_class_num,
            bias=False)

        self.decoder.apply(weights_init_classifier)

    def forward(self, inputs: Dict[str, torch.Tensor],
                train_mode: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        input_anchors = {}
        if train_mode:
            input_positives = {}
            input_negatives = {}
            for modality in self.cfg.model_modalities:
                input_anchors[modality] = inputs[modality][:, 0]
                input_positives[modality] = inputs[modality][:, 1]
                input_negatives[modality] = inputs[modality][:, 2]
        else:
            for modality in self.cfg.model_modalities:
                input_anchors[modality] = inputs[modality]

        z_anchor = []
        for modality in self.cfg.model_modalities:
            # print("98 raw input shape, b, s, d", input_anchors[modality].shape)
            z_anchor.append(self.transformer(input_anchors[modality]).last_hidden_state.permute(1, 0, 2)) 
            # print("100 after transformer s, b, d", z_anchor[-1].shape)

        z_anchor = torch.cat(z_anchor, dim=0)
        # print("104 after concat s*3, b, d", z_anchor.shape)

        anchor_output = self.vanilla_fusion_transformers(z_anchor) # [s, b, d]
        if self.cfg.avg_output_tokens:
            anchor_output = torch.mean(anchor_output, dim=0)
        else:
            anchor_output = anchor_output[self.cfg.token_step_to_decode, :, :] # [b, d], flattens automatically

        output_embeddings = {}
        output_embeddings["z_reparamed_anchor"] = anchor_output

        if train_mode:
            z_pos = []
            for modality in self.cfg.model_modalities:
                z_pos.append(self.transformer(input_positives[modality]).last_hidden_state.permute(1, 0, 2))
            
            z_pos = torch.cat(z_pos, dim=0)
            pos_output = self.vanilla_fusion_transformers(z_pos)
            if self.cfg.avg_output_tokens:
                pos_output = torch.mean(pos_output, dim=0)
            else:
                pos_output = pos_output[self.cfg.token_step_to_decode, :, :]

            z_neg = []
            for modality in self.cfg.model_modalities:
                z_neg.append(self.transformer(input_negatives[modality]).last_hidden_state.permute(1, 0, 2))

            z_neg = torch.cat(z_neg, dim=0)
            neg_output = self.vanilla_fusion_transformers(z_neg)
            if self.cfg.avg_output_tokens:
                neg_output = torch.mean(neg_output, dim=0)
            else:
                neg_output = neg_output[self.cfg.token_step_to_decode, :, :]
                

            output_embeddings["z_reparamed_positive"] = pos_output
            output_embeddings["z_reparamed_negative"] = neg_output
            
            anchor_output = self.bottleneck(anchor_output)
            output_class = self.decoder(anchor_output)

            return output_class, output_embeddings

        return output_embeddings
