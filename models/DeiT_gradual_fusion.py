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


class DEIT_Gradual_Fusion(nn.Module):
    def __init__(self,
                 cfg: Dict[str, Union[int, str]], 
                 fabric: any) -> None:
        super(DEIT_Gradual_Fusion, self).__init__()
        self.cfg = cfg
        self.fabric = fabric
        hidden_size = self.cfg.vit_embed_dim
        self.feat_dim = (self.cfg.vit_embed_dim * len(self.cfg.model_modalities) * 
                         self.cfg.model_num_cls_tokens) + self.cfg.model_num_fusion_tokens * self.cfg.vit_embed_dim # data + fusion token

        self.cls_anchor: ParameterDict = nn.ParameterDict()
        self.modality_transformers: ModuleDict = nn.ModuleDict()

        for modality in self.cfg.model_modalities:
            self.cls_anchor[modality] = nn.Parameter(
                    nn.init.xavier_uniform_(
                        torch.empty(self.cfg.model_num_cls_tokens, 1, hidden_size)
                                        ).to(self.fabric.device))
            self.modality_transformers[modality] = nn.TransformerEncoderLayer(d_model=hidden_size,
                                        nhead=self.cfg.model_num_heads).to(self.fabric.device)

        self.fusion_tokens = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty(self.cfg.model_num_fusion_tokens, 1,
                                hidden_size)).to(self.fabric.device))
        
        self.fusion_avg_params = nn.Parameter(self.fabric.to_device(
                                                torch.full((len(self.cfg.model_modalities),), 1/len(self.cfg.model_modalities)) 
                                                ), requires_grad=(self.cfg.model_lrnable_fusion_avg) )

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

        z_anchors = {}
        z_anchors_joint = {}
        fusion_anchor_output = []
        anchor_output = []
        cls_anchor = {}
        for modality in self.cfg.model_modalities:

            # pass through pretrained encoder
            z_anchors[modality] = self.transformer(input_anchors[modality])

            # (B, S, D) -> (S, B, D)
            z_anchors[modality] = z_anchors[modality].last_hidden_state.permute(1, 0, 2)

            # create modality tokens
            cls_anchor[modality] = self.cls_anchor[modality].repeat(1, z_anchors[modality].shape[1], 1) # (seq_len, batch, dim)

            # concat modality, fusion, and data tokens
            if self.cfg.model_fusion_combos[0] == "f":
                z_anchors[modality] = torch.cat(
                    (cls_anchor[modality],
                     self.fusion_tokens.repeat(1, z_anchors[modality].shape[1], 1), 
                     z_anchors[modality]),
                    dim=0)
            elif self.cfg.model_fusion_combos[0] == "d":
                z_anchors[modality] = torch.cat(
                    (cls_anchor[modality], 
                     z_anchors[modality],
                     self.fusion_tokens.repeat(1, z_anchors[modality].shape[1], 1)),
                    dim=0)

            # pass through individual modality transformers
            z_anchors_joint[modality] = self.modality_transformers[modality](z_anchors[modality])

            # (S, B, D) -> (B, S, D)
            z_anchors_joint[modality] = z_anchors_joint[modality].permute(1, 0, 2)
            cls_anchor[modality] = cls_anchor[modality].permute(1, 0, 2) # (batch, seq_len, dim)

            # extract fusion tokens
            fusion_anchor = z_anchors_joint[modality][:, self.cfg.model_num_cls_tokens:self.cfg.model_num_cls_tokens + self.cfg.model_num_fusion_tokens, :]
            
            if not self.cfg.lagging_modality_token:
                    cls_anchor[modality] = z_anchors_joint[modality][:, :self.cfg.model_num_cls_tokens, :]

            # flatten 
            cls_anchor_flattened = cls_anchor[modality].reshape(cls_anchor[modality].shape[0], -1)
            fusion_anchor_flattened = fusion_anchor.reshape(fusion_anchor.shape[0], -1)
            anchor_output.append(cls_anchor_flattened)
            fusion_anchor_output.append(fusion_anchor_flattened)

        fusion_anchor_output = torch.stack(fusion_anchor_output) # (modality, batch, dim)
        fusion_anchor_output = (self.fusion_avg_params.unsqueeze(1).unsqueeze(2) * fusion_anchor_output).sum(0) # expand into in batch & dim, sum across modality

        anchor_output.append(fusion_anchor_output)
        anchor_output = torch.cat(anchor_output, dim=1)

        output_embeddings = {}
        output_embeddings["z_reparamed_anchor"] = anchor_output

        if train_mode:
            z_pos = {}
            z_pos_joint = {}
            fusion_pos_output = []
            pos_output = []
            cls_pos = {}
            for modality in self.cfg.model_modalities:
            
                # pass through pretrained encoder
                z_pos[modality] = self.transformer(input_positives[modality])

                # (B, S, D) -> (S, B, D)
                z_pos[modality] = z_pos[modality].last_hidden_state.permute(
                    1, 0, 2)
                
                # create modality tokens
                cls_pos[modality] = self.cls_anchor[modality].repeat(
                    1, z_pos[modality].shape[1], 1)
                if self.cfg.model_fusion_combos[1] == "f":
                    z_pos[modality] = torch.cat(
                        (cls_pos[modality],
                         self.fusion_tokens.repeat(1, z_pos[modality].shape[1], 1), z_pos[modality]),
                        dim=0)
                elif self.cfg.model_fusion_combos[1] == "d":
                    z_pos[modality] = torch.cat(
                        (cls_pos[modality], 
                         z_pos[modality],
                         self.fusion_tokens.repeat(1, z_pos[modality].shape[1], 1)),
                        dim=0)

                # pass through individual modality transformers
                z_pos_joint[modality] = self.modality_transformers[modality](z_pos[modality])

                # (S, B, D) -> (B, S, D)
                z_pos_joint[modality] = z_pos_joint[modality].permute(1, 0, 2)
                cls_pos[modality] = cls_pos[modality].permute(1, 0, 2)

                # extract fusion tokens
                fusion_pos = z_pos_joint[
                    modality][:, self.cfg.model_num_cls_tokens:self.cfg.model_num_cls_tokens +
                              self.cfg.model_num_fusion_tokens, :]
                if not self.cfg.lagging_modality_token:
                    cls_pos[modality] = z_pos_joint[modality][:, :self.cfg.model_num_cls_tokens, :]

                # flatten
                cls_pos_flattened = cls_pos[modality].reshape(cls_pos[modality].shape[0], -1)
                fusion_pos_flattened = fusion_pos.reshape(fusion_pos.shape[0], -1)
                pos_output.append(cls_pos_flattened)
                fusion_pos_output.append(fusion_pos_flattened)
            fusion_pos_output = torch.stack(fusion_pos_output)
            fusion_pos_output = (self.fusion_avg_params.unsqueeze(1).unsqueeze(2) * fusion_pos_output).sum(0)

            pos_output.append(fusion_pos_output)
            pos_output = torch.cat(pos_output, dim=1)
                
            z_neg = {}
            z_neg_joint = {}
            fusion_neg_output = []
            neg_output = []
            cls_neg = {}
            for modality in self.cfg.model_modalities:

                # pass through pretrained encoder
                z_neg[modality] = self.transformer(input_negatives[modality])

                # (B, S, D) -> (S, B, D)
                z_neg[modality] = z_neg[modality].last_hidden_state.permute(
                    1, 0, 2)

                # create modality tokens
                cls_neg[modality] = self.cls_anchor[modality].repeat(
                    1, z_neg[modality].shape[1], 1)

                # concat modality, fusion, and data tokens
                if self.cfg.model_fusion_combos[2] == "f":
                    z_neg[modality] = torch.cat(
                        (cls_neg[modality],
                         self.fusion_tokens.repeat(1, z_neg[modality].shape[1], 1), 
                         z_neg[modality]),
                        dim=0)
                elif self.cfg.model_fusion_combos[2] == "d":
                    z_neg[modality] = torch.cat(
                        (cls_neg[modality], 
                         z_neg[modality],
                         self.fusion_tokens.repeat(1, z_neg[modality].shape[1], 1)),
                        dim=0)

                # pass through individual modality transformers
                z_neg_joint[modality] = self.modality_transformers[modality](
                    z_neg[modality])

                # (S, B, D) -> (B, S, D)
                z_neg_joint[modality] = z_neg_joint[modality].permute(
                    1, 0, 2)
                cls_neg[modality] = cls_neg[modality].permute(1, 0, 2)

                # extract fusion tokens
                fusion_neg = z_neg_joint[
                    modality][:, self.cfg.model_num_cls_tokens +
                                self.cfg.data_token_step:self.cfg.model_num_cls_tokens +
                                self.cfg.model_num_fusion_tokens +
                                self.cfg.data_token_step, :]     
                           
                if not self.cfg.lagging_modality_token:
                    cls_neg[modality] = z_neg_joint[modality][:, :self.cfg.model_num_cls_tokens, :]
                
                # flatten
                fusion_neg_flattened = fusion_neg.reshape(
                    fusion_neg.shape[0], -1)
                cls_neg_flattened = cls_neg[modality].reshape(
                    cls_neg[modality].shape[0], -1)
                neg_output.append(cls_neg_flattened)
                fusion_neg_output.append(fusion_neg_flattened)

            fusion_neg_output = torch.stack(fusion_neg_output)
            fusion_neg_output = (self.fusion_avg_params.unsqueeze(1).unsqueeze(2) * fusion_neg_output).sum(0)

            neg_output.append(fusion_neg_output)
            neg_output = torch.cat(neg_output, dim=1)

            output_embeddings["z_reparamed_positive"] = pos_output
            output_embeddings["z_reparamed_negative"] = neg_output
            
            anchor_output = self.bottleneck(anchor_output)
            output_class = self.decoder(anchor_output)

            return output_class, output_embeddings

        return output_embeddings
