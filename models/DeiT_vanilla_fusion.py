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
        self.feat_dim = self.cfg.vit_embed_dim * 198# * len(self.cfg.model_modalities)

        self.modality_transformer = self.fabric.to_device(
            nn.TransformerEncoderLayer(d_model=hidden_size,
                                        nhead=self.cfg.model_num_heads))
    
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
        output_embeddings = {}

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

        z_anchors = []
        for modality in self.cfg.model_modalities:
            z_temp = self.transformer(input_anchors[modality]).last_hidden_state.permute(1, 0, 2)
            z_anchors.append(z_temp)

        z_anchors = torch.cat(z_anchors, dim=0)
        fusion_anchor_output = self.modality_transformer(z_anchors).permute(1, 0, 2)

        fusion_anchor_output_chunked = torch.chunk(fusion_anchor_output, chunks=len(self.cfg.model_modalities), dim=1)
        fusion_anchor_output = torch.stack(fusion_anchor_output_chunked, dim=0)
        fusion_anchor_output_mean = fusion_anchor_output.mean(dim=0) #.permute(1, 0, 2, 3) #.mean(dim=0)
        # print("fusion_anchor_output_mean", fusion_anchor_output_mean.size())

        fusion_anchor_output = fusion_anchor_output_mean.reshape(fusion_anchor_output_mean.size(0), -1)
        # fusion_anchor_output = torch.mean(fusion_anchor_output)

        output_embeddings["z_reparamed_anchor"] = fusion_anchor_output

        if train_mode:

            z_positives = []
            for modality in self.cfg.model_modalities:
                z_positives.append(self.transformer(input_positives[modality]).last_hidden_state.permute(1, 0, 2))
            z_positives = torch.cat(z_positives, dim=0)
            fusion_positives_output = self.modality_transformer(z_positives).permute(1, 0, 2)
            fusion_positives_output_chunked = torch.chunk(fusion_positives_output, chunks=len(self.cfg.model_modalities), dim=1)
            fusion_positives_output = torch.stack(fusion_positives_output_chunked, dim=0)
            fusion_positives_output_mean = fusion_positives_output.mean(dim=0) #.permute(1, 0, 2, 3)#.mean(dim=0)

            fusion_positives_output = fusion_positives_output_mean.reshape(fusion_positives_output_mean.size(0), -1)
            # fusion_anchor_output = torch.mean(fusion_anchor_output)

            # fusion_positives_output = fusion_positives_output.reshape(fusion_positives_output.size(0), -1)

            output_embeddings["z_reparamed_positive"] = fusion_positives_output

            z_negatives = []
            for modality in self.cfg.model_modalities:
                z_negatives.append(self.transformer(input_negatives[modality]).last_hidden_state.permute(1, 0, 2))
            z_negatives = torch.cat(z_negatives, dim=0)
            fusion_negatives_output = self.modality_transformer(z_negatives).permute(1, 0, 2)
            
            fusion_negatives_output_chunked = torch.chunk(fusion_negatives_output, chunks=len(self.cfg.model_modalities), dim=1)
            fusion_negatives_output = torch.stack(fusion_negatives_output_chunked, dim=0)
            fusion_negatives_output_mean = fusion_negatives_output.mean(dim=0) #.permute(1, 0, 2, 3)#.mean(dim=0)

            fusion_negatives_output = fusion_negatives_output_mean.reshape(fusion_negatives_output_mean.size(0), -1)

            output_embeddings["z_reparamed_negative"] = fusion_negatives_output

            fusion_anchor_output = self.bottleneck(fusion_anchor_output)
            output_class = self.decoder(fusion_anchor_output)

            return output_class, output_embeddings

        return output_embeddings
