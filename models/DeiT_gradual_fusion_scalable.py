import torch
import torch.nn as nn
from transformers import DeiTModel
import torch.distributed as dist
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


class DEIT_Gradual_Fusion_Scalable(nn.Module):
    def __init__(self, 
                 cfg: Dict[str, Union[int, str]], 
                 fabric: any, 
                 process_group: Optional[dist.ProcessGroup] = None) -> None:
        super(DEIT_Gradual_Fusion_Scalable, self).__init__()
        self.cfg = cfg
        self.fabric = fabric
        self.process_group = process_group
        hidden_size = self.cfg.vit_embed_dim
        self.feat_dim = self.cfg.vit_embed_dim * (
            len(self.cfg.model_modalities) *
            self.cfg.model_num_cls_tokens +
            self.cfg.model_num_fusion_tokens)

        self.cls_anchor: ParameterDict = nn.ParameterDict()
        self.modality_transformers: ModuleDict = nn.ModuleDict()

        for modality in self.cfg.model_modalities:
            self.cls_anchor[modality] = self.fabric.to_device(
                nn.Parameter(
                    nn.init.xavier_uniform_(
                        torch.empty(self.cfg.model_num_cls_tokens, 1,
                                    hidden_size))))
            self.modality_transformers[modality] = self.fabric.to_device(
                nn.Sequential(
                    nn.TransformerEncoderLayer(d_model=hidden_size,
                                           nhead=self.cfg.model_num_heads, 
                                           dim_feedforward=cfg.model_dim_feedforward)))
            
            for i in range(self.cfg.model_num_transformer_layers - 1):
                self.modality_transformers[modality].add_module(
                    f"transformer_encoder_layer_{i}",
                    nn.TransformerEncoderLayer(d_model=hidden_size,
                                               nhead=self.cfg.model_num_heads, 
                                               dim_feedforward=cfg.model_dim_feedforward))

        self.fusion_tokens = self.fabric.to_device(
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty(self.cfg.model_num_fusion_tokens, 1,
                                hidden_size))))

        print("Loading pretrained transformer...")

        if self.cfg.pretrained_model == "distilled-384": 
            self.transformer = DeiTModel.from_pretrained(
                'facebook/deit-base-distilled-patch16-384')
        elif self.cfg.pretrained_model == "distilled-224":
            self.transformer = DeiTModel.from_pretrained(
                'facebook/deit-base-distilled-patch16-224')
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
                self.bottleneck, process_group=process_group)

        self.decoder = nn.Linear(
            (self.cfg.model_num_fusion_tokens +
             self.cfg.model_num_cls_tokens *
             len(self.cfg.model_modalities)) * hidden_size,
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

        z_anchors = {}
        z_anchors_joint = {}
        fusion_anchor_output = []
        anchor_output = []
        cls_anchor = {}
        for modality in self.cfg.model_modalities:
            z_anchors[modality] = self.transformer(input_anchors[modality])
            z_anchors[modality] = z_anchors[modality].last_hidden_state.permute(1, 0, 2)
            cls_anchor[modality] = self.cls_anchor[modality].repeat(1, z_anchors[modality].shape[1], 1)
            if self.cfg.model_fusion_combos[0] == "f":
                z_anchors[modality] = torch.cat(
                    (cls_anchor[modality],
                     self.fusion_tokens.repeat(1, z_anchors[modality].shape[1], 1), z_anchors[modality]),
                    dim=0)
            elif self.cfg.model_fusion_combos[0] == "d":
                z_anchors[modality] = torch.cat(
                    (cls_anchor[modality], z_anchors[modality],
                     self.fusion_tokens.repeat(1, z_anchors[modality].shape[1], 1)),
                    dim=0)
            z_anchors_joint[modality] = self.modality_transformers[modality](
                z_anchors[modality])
            z_anchors_joint[modality] = z_anchors_joint[modality].permute(
                1, 0, 2)
            cls_anchor[modality] = cls_anchor[modality].permute(1, 0, 2)
            fusion_anchor = z_anchors_joint[
                modality][:, self.cfg.model_num_cls_tokens:self.cfg.model_num_cls_tokens +
                          self.cfg.model_num_fusion_tokens, :]
            cls_anchor_flattened = cls_anchor[modality].reshape(
                cls_anchor[modality].shape[0], -1)
            fusion_anchor_flattened = fusion_anchor.reshape(
                fusion_anchor.shape[0], -1)
            anchor_output.append(cls_anchor_flattened)
            fusion_anchor_output.append(fusion_anchor_flattened)

        fusion_anchor_output = torch.mean(torch.stack(fusion_anchor_output),
                                          dim=0)

        if self.cfg.model_anchor_only_reid:
            anchor_reid = fusion_anchor_output
            anchor_output.append(fusion_anchor_output)
            anchor_output = torch.cat(anchor_output, dim=1)
        else:
            anchor_output.append(fusion_anchor_output)
            anchor_output = torch.cat(anchor_output, dim=1)

        output_embeddings = {}
        if self.cfg.model_anchor_only_reid:
            output_embeddings["z_reparamed_anchor"] = anchor_reid
        else:
            output_embeddings["z_reparamed_anchor"] = anchor_output

        if train_mode:
            z_pos = {}
            z_pos_joint = {}
            fusion_pos_output = []
            pos_output = []
            cls_pos = {}
            for modality in self.cfg.model_modalities:
                z_pos[modality] = self.transformer(input_positives[modality])
                z_pos[modality] = z_pos[modality].last_hidden_state.permute(
                    1, 0, 2)
                cls_pos[modality] = self.cls_anchor[modality].repeat(
                    1, z_pos[modality].shape[1], 1)
                if self.cfg.model_fusion_combos[1] == "f":
                    z_pos[modality] = torch.cat(
                        (cls_pos[modality],
                         self.fusion_tokens.repeat(1, z_pos[modality].shape[1], 1), z_pos[modality]),
                        dim=0)
                elif self.cfg.model_fusion_combos[1] == "d":
                    z_pos[modality] = torch.cat(
                        (cls_pos[modality], z_pos[modality],
                         self.fusion_tokens.repeat(1, z_pos[modality].shape[1], 1)),
                        dim=0)
                z_pos_joint[modality] = self.modality_transformers[modality](
                    z_pos[modality])
                z_pos_joint[modality] = z_pos_joint[modality].permute(
                    1, 0, 2)
                cls_pos[modality] = cls_pos[modality].permute(1, 0, 2)
                fusion_pos = z_pos_joint[
                    modality][:, self.cfg.model_num_cls_tokens:self.cfg.model_num_cls_tokens +
                              self.cfg.model_num_fusion_tokens, :]
                cls_pos_flattened = cls_pos[modality].reshape(
                    cls_pos[modality].shape[0], -1)
                fusion_pos_flattened = fusion_pos.reshape(
                    fusion_pos.shape[0], -1)
                pos_output.append(cls_pos_flattened)
                fusion_pos_output.append(fusion_pos_flattened)
            fusion_pos_output = torch.mean(torch.stack(fusion_pos_output),
                                           dim=0)

            if self.cfg.model_anchor_only_reid:
                pos_reid = fusion_pos_output
                pos_output.append(fusion_pos_output)
                pos_output = torch.cat(pos_output, dim=1)
            else:
                pos_output.append(fusion_pos_output)
                pos_output = torch.cat(pos_output, dim=1)

            z_neg = {}
            z_neg_joint = {}
            fusion_neg_output = []
            neg_output = []
            cls_neg = {}
            for modality in self.cfg.model_modalities:
                z_neg[modality] = self.transformer(input_negatives[modality])
                z_neg[modality] = z_neg[modality].last_hidden_state.permute(
                    1, 0, 2)
                cls_neg[modality] = self.cls_anchor[modality].repeat(
                    1, z_neg[modality].shape[1], 1)
                if self.cfg.model_fusion_combos[2] == "f":
                    z_neg[modality] = torch.cat(
                        (cls_neg[modality],
                         self.fusion_tokens.repeat(1, z_neg[modality].shape[1], 1), z_neg[modality]),
                        dim=0)
                elif self.cfg.model_fusion_combos[2] == "d":
                    z_neg[modality] = torch.cat(
                        (cls_neg[modality], z_neg[modality],
                         self.fusion_tokens.repeat(1, z_neg[modality].shape[1], 1)),
                        dim=0)
                z_neg_joint[modality] = self.modality_transformers[modality](
                    z_neg[modality])
                z_neg_joint[modality] = z_neg_joint[modality].permute(
                    1, 0, 2)
                cls_neg[modality] = cls_neg[modality].permute(1, 0, 2)
                cls_neg_flattened = cls_neg[modality].reshape(
                    cls_neg[modality].shape[0], -1)
                if self.cfg.model_fusion_combos[2] == "f":
                    fusion_neg = z_neg_joint[
                        modality][:, self.cfg.model_num_cls_tokens:self.cfg.model_num_cls_tokens +
                                  self.cfg.model_num_fusion_tokens, :]
                else:
                    fusion_neg = z_neg_joint[
                        modality][:, self.cfg.model_num_cls_tokens +
                                  self.cfg.data_token_step:self.cfg.model_num_cls_tokens +
                                  self.cfg.model_num_fusion_tokens +
                                  self.cfg.data_token_step, :]
                fusion_neg_flattened = fusion_neg.reshape(
                    fusion_neg.shape[0], -1)
                neg_output.append(cls_neg_flattened)
                fusion_neg_output.append(fusion_neg_flattened)
            fusion_neg_output = torch.mean(torch.stack(fusion_neg_output),
                                           dim=0)

            if self.cfg.model_anchor_only_reid:
                neg_reid = fusion_neg_output
                neg_output.append(fusion_neg_output)
                neg_output = torch.cat(neg_output, dim=1)
            else:
                neg_output.append(fusion_neg_output)
                neg_output = torch.cat(neg_output, dim=1)

            if self.cfg.model_anchor_only_reid:
                output_embeddings["z_reparamed_positive"] = pos_reid
                output_embeddings["z_reparamed_negative"] = neg_reid
            else:
                output_embeddings["z_reparamed_positive"] = pos_output
                output_embeddings["z_reparamed_negative"] = neg_output

            anchor_output = self.bottleneck(anchor_output)
            output_class = self.decoder(anchor_output)

            return output_class, output_embeddings

        return output_embeddings
