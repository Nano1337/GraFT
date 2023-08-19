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


class DEIT_Gradual_Fusion_V4(nn.Module):
    def __init__(self,
                 cfg: Dict[str, Union[int, str]], 
                 fabric: any) -> None:
        super(DEIT_Gradual_Fusion_V4, self).__init__()
        self.cfg = cfg
        self.fabric = fabric
        hidden_size = self.cfg.vit_embed_dim
        self.feat_dim = 256 * len(self.cfg.model_modalities) + (self.cfg.model_num_fusion_tokens * hidden_size)
            # self.cfg.vit_embed_dim * (
            # len(self.cfg.model_modalities) *
            # self.cfg.model_num_cls_tokens +
            # self.cfg.model_num_fusion_tokens)
        
        conv_p = [[1, 4, 16, 4],
                [4, 8, 13, 4],
                [8, 8, 13, 4]]
        self.data_token_dim = 256

        self.cls_anchor: ParameterDict = nn.ParameterDict()
        self.modality_transformers: ModuleDict = nn.ModuleDict()
        self.conv_pool_embedding: ModuleDict = nn.ModuleDict() # downsample enbedding dimension
        self.fc_downsample_data_toks: ModuleDict = nn.ModuleDict() # downsample tokens

        #new
        for modality in self.cfg.model_modalities:
            self.cls_anchor[modality] = nn.Parameter(
                    nn.init.xavier_uniform_(
                        torch.empty(self.cfg.model_num_cls_tokens, 1,
                                    hidden_size)).to(self.fabric.device))
            self.modality_transformers[modality] = nn.TransformerEncoderLayer(d_model=hidden_size,
                                        nhead=self.cfg.model_num_heads).to(self.fabric.device)
            
            self.conv_pool_embedding[modality] = nn.Sequential(
                nn.Conv1d(in_channels=conv_p[0][0], out_channels=conv_p[0][1], kernel_size=conv_p[0][2], stride=conv_p[0][3]),
                nn.BatchNorm1d(conv_p[0][1]),
                # nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=conv_p[1][0], out_channels=conv_p[1][1], kernel_size=conv_p[1][2], stride=conv_p[1][3]),
                nn.BatchNorm1d(conv_p[1][1]),
                # nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=conv_p[2][0], out_channels=conv_p[2][1], kernel_size=conv_p[2][2], stride=conv_p[2][3]),
                nn.BatchNorm1d(conv_p[2][1]),
                # nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=3)
            )
            self.fc_downsample_data_toks[modality] = nn.Linear(in_features=4752, out_features=self.data_token_dim, bias=True) 

            

        self.fusion_tokens = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty(self.cfg.model_num_fusion_tokens, 1,
                                hidden_size)).to(self.fabric.device))
        
        
        self.fusion_avg_params = nn.Parameter(self.fabric.to_device(
                                                torch.full((len(self.cfg.model_modalities),), 1/len(self.cfg.model_modalities)) 
                                                ), requires_grad=(self.cfg.train_stage != 1) )
        

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
        self.transformer.eval()
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
            self.feat_dim, #768 + 3*256
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
        data_anchor_output = [] #new
        data_anchor = {} # data tokens
        for modality in self.cfg.model_modalities:
            # pass through pretrained encoder
            z_anchors[modality] = self.transformer(input_anchors[modality])
            # (B, S, D) -> (S, B, D)
            z_anchors[modality] = z_anchors[modality].last_hidden_state.permute(1, 0, 2)
            # create modality tokens
            cls_anchor[modality] = self.cls_anchor[modality].repeat(1, z_anchors[modality].shape[1], 1) # (seq_len, batch, dim)

            # concat modality, fusion, and data tokens
            z_anchors[modality] = torch.cat(
                    (cls_anchor[modality],
                     self.fusion_tokens.repeat(1, z_anchors[modality].shape[1], 1), 
                     z_anchors[modality]),
                    dim=0)

            # pass through individual modality transformers
            z_anchors_joint[modality] = self.modality_transformers[modality](z_anchors[modality]) 
            z_anchors_joint[modality] = z_anchors_joint[modality].permute(1, 0, 2) # (S, B, D) -> (B, S, D)

            # extract fusion tokens & modality & data tokens
            fusion_anchor = z_anchors_joint[
                modality][:, self.cfg.model_num_cls_tokens:self.cfg.model_num_cls_tokens +
                          self.cfg.model_num_fusion_tokens, :]
            cls_anchor[modality] = cls_anchor[modality].permute(1, 0, 2) # (batch, seq_len, dim)
            if not self.cfg.lagging_modality_token:
                print("plucking out modality tokens")
                cls_anchor[modality] = z_anchors_joint[modality][:, :self.cfg.model_num_cls_tokens, :]
            data_anchor[modality] = z_anchors_joint[modality][:, self.cfg.model_num_cls_tokens + self.cfg.model_num_fusion_tokens:, :]
            # print(data_anchor[modality].shape, "data_anchor should be B=20, S=197, D=768")

            # compress data tokens
            B, S, D = data_anchor[modality].shape
            data_anchor[modality] = data_anchor[modality].reshape(B*S, -1).unsqueeze(1) # (B, S, D) -> (B*S, D) -> (B*S, 1, D)
            data_anchor[modality] = self.conv_pool_embedding[modality](data_anchor[modality]) # (B*S, 1, D) -> (B*S, out_channel=8, dim=3)
            data_anchor[modality] = data_anchor[modality].reshape(B, -1) # (B*S, out_channel=8, dim=3) -> (B, S*8*3=197*8*3=4728)
            data_anchor[modality] = self.fc_downsample_data_toks[modality](data_anchor[modality]) # (B, 4728) -> (B, 256)
            data_anchor_output.append(data_anchor[modality]) # (B, 256)

            # flatten & concat cls, fusion, and data tokens
            cls_anchor_flattened = cls_anchor[modality].reshape(cls_anchor[modality].shape[0], -1)
            fusion_anchor_flattened = fusion_anchor.reshape(fusion_anchor.shape[0], -1)
            anchor_output.append(cls_anchor_flattened)
            fusion_anchor_output.append(fusion_anchor_flattened)
            # print(cls_anchor_flattened.shape, "cls_anchor should be B=20, seq=1*dim=768")

        fusion_anchor_output = torch.stack(fusion_anchor_output) # (modality, batch, dim) -> 
        fusion_anchor_output = (self.fusion_avg_params.unsqueeze(1).unsqueeze(2) * fusion_anchor_output).sum(0) # expand into in batch & dim, sum across modality

        data_anchor_output = torch.stack(data_anchor_output).permute(1, 0, 2) # (modality, batch, dim) -> (batch, modality, dim)
        data_anchor_output = data_anchor_output.reshape(data_anchor_output.shape[0], -1) # (batch, modality, dim) -> (batch, modality*dim)

        anchor_output.append(fusion_anchor_output) 
        anchor_output.append(data_anchor_output) #
        anchor_output = torch.cat(anchor_output, dim=1)
        # print(anchor_output.shape, " should be B=20, seq=(3*256)+768")

        output_embeddings = {}
        
        output_embeddings["z_reparamed_anchor"] = anchor_output

        if train_mode:
            z_positives = {}
            z_positives_joint = {}
            fusion_positive_output = []
            positive_output = []
            cls_positive = {}
            data_positive_output = [] #new
            data_positive = {} # data tokens
            for modality in self.cfg.model_modalities:
                # pass through pretrained encoder
                z_positives[modality] = self.transformer(input_positives[modality])
                # (B, S, D) -> (S, B, D)
                z_positives[modality] = z_positives[modality].last_hidden_state.permute(1, 0, 2)
                # create modality tokens
                cls_positive[modality] = self.cls_anchor[modality].repeat(1, z_positives[modality].shape[1], 1) # (seq_len, batch, dim)

                # concat modality, fusion, and data tokens
                z_positives[modality] = torch.cat(
                        (cls_positive[modality],
                        self.fusion_tokens.repeat(1, z_positives[modality].shape[1], 1), 
                        z_positives[modality]),
                        dim=0)

                # pass through individual modality transformers
                z_positives_joint[modality] = self.modality_transformers[modality](z_positives[modality]) 
                z_positives_joint[modality] = z_positives_joint[modality].permute(1, 0, 2) # (S, B, D) -> (B, S, D)

                # extract fusion tokens & modality & data tokens
                fusion_positive = z_positives_joint[
                    modality][:, self.cfg.model_num_cls_tokens:self.cfg.model_num_cls_tokens +
                            self.cfg.model_num_fusion_tokens, :]
                cls_positive[modality] = cls_positive[modality].permute(1, 0, 2) # (batch, seq_len, dim)
                if not self.cfg.lagging_modality_token:
                    print("plucking out modality tokens")
                    cls_positive[modality] = z_positives_joint[modality][:, :self.cfg.model_num_cls_tokens, :]
                data_positive[modality] = z_positives_joint[modality][:, self.cfg.model_num_cls_tokens + self.cfg.model_num_fusion_tokens:, :]
                # print(data_positive[modality].shape, "data_positive should be B=20, S=197, D=768")

                # compress data tokens
                B, S, D = data_positive[modality].shape
                data_positive[modality] = data_positive[modality].reshape(B*S, -1).unsqueeze(1) # (B, S, D) -> (B*S, D) -> (B*S, 1, D)
                data_positive[modality] = self.conv_pool_embedding[modality](data_positive[modality]) # (B*S, 1, D) -> (B*S, out_channel=8, dim=3)
                data_positive[modality] = data_positive[modality].reshape(B, -1) # (B*S, out_channel=8, dim=3) -> (B, S*8*3=197*8*3=4728)
                data_positive[modality] = self.fc_downsample_data_toks[modality](data_positive[modality]) # (B, 4728) -> (B, 256)
                data_positive_output.append(data_positive[modality]) # (B, 256)

                # flatten & concat cls, fusion, and data tokens
                cls_positive_flattened = cls_positive[modality].reshape(cls_positive[modality].shape[0], -1)
                fusion_positive_flattened = fusion_positive.reshape(fusion_positive.shape[0], -1)
                positive_output.append(cls_positive_flattened)
                fusion_positive_output.append(fusion_positive_flattened)
                # print(cls_positive_flattened.shape, "cls_positive should be B=20, seq=1*dim=768")

            fusion_positive_output = torch.stack(fusion_positive_output) # (modality, batch, dim) -> 
            fusion_positive_output = (self.fusion_avg_params.unsqueeze(1).unsqueeze(2) * fusion_positive_output).sum(0) # expand into in batch & dim, sum across modality

            data_positive_output = torch.stack(data_positive_output).permute(1, 0, 2) # (modality, batch, dim) -> (batch, modality, dim)
            data_positive_output = data_positive_output.reshape(data_positive_output.shape[0], -1) # (batch, modality, dim) -> (batch, modality*dim)

            positive_output.append(fusion_positive_output) 
            positive_output.append(data_positive_output) #
            positive_output = torch.cat(positive_output, dim=1)
            # print(positive_output.shape, " should be B=20, seq=(3*256)+768")

            output_embeddings["z_reparamed_positive"] = positive_output

            # Negatives
            z_negatives = {}
            z_negatives_joint = {}
            fusion_negative_output = []
            negative_output = []
            cls_negative = {}
            data_negative_output = [] #new
            data_negative = {} # data tokens
            for modality in self.cfg.model_modalities:
                # pass through pretrained encoder
                z_negatives[modality] = self.transformer(input_negatives[modality])
                # (B, S, D) -> (S, B, D)
                z_negatives[modality] = z_negatives[modality].last_hidden_state.permute(1, 0, 2)
                # create modality tokens
                cls_negative[modality] = self.cls_anchor[modality].repeat(1, z_negatives[modality].shape[1], 1) # (seq_len, batch, dim)

                # concat modality, fusion, and data tokens
                z_negatives[modality] = torch.cat(
                        (cls_negative[modality],
                        self.fusion_tokens.repeat(1, z_negatives[modality].shape[1], 1), 
                        z_negatives[modality]),
                        dim=0)

                # pass through individual modality transformers
                z_negatives_joint[modality] = self.modality_transformers[modality](z_negatives[modality]) 
                z_negatives_joint[modality] = z_negatives_joint[modality].permute(1, 0, 2) # (S, B, D) -> (B, S, D)

                # extract fusion tokens & modality & data tokens
                fusion_negative = z_negatives_joint[
                    modality][:, self.cfg.model_num_cls_tokens:self.cfg.model_num_cls_tokens +
                            self.cfg.model_num_fusion_tokens, :]
                cls_negative[modality] = cls_negative[modality].permute(1, 0, 2) # (batch, seq_len, dim)
                if not self.cfg.lagging_modality_token:
                    print("plucking out modality tokens")
                    cls_negative[modality] = z_negatives_joint[modality][:, :self.cfg.model_num_cls_tokens, :]
                data_negative[modality] = z_negatives_joint[modality][:, self.cfg.model_num_cls_tokens + self.cfg.model_num_fusion_tokens:, :]
                # print(data_negative[modality].shape, "data_negative should be B=20, S=197, D=768")

                # compress data tokens
                B, S, D = data_negative[modality].shape
                data_negative[modality] = data_negative[modality].reshape(B*S, -1).unsqueeze(1) # (B, S, D) -> (B*S, D) -> (B*S, 1, D)
                data_negative[modality] = self.conv_pool_embedding[modality](data_negative[modality]) # (B*S, 1, D) -> (B*S, out_channel=8, dim=3)
                data_negative[modality] = data_negative[modality].reshape(B, -1) # (B*S, out_channel=8, dim=3) -> (B, S*8*3=197*8*3=4728)
                data_negative[modality] = self.fc_downsample_data_toks[modality](data_negative[modality]) # (B, 4728) -> (B, 256)
                data_negative_output.append(data_negative[modality]) # (B, 256)

                # flatten & concat cls, fusion, and data tokens
                cls_negative_flattened = cls_negative[modality].reshape(cls_negative[modality].shape[0], -1)
                fusion_negative_flattened = fusion_negative.reshape(fusion_negative.shape[0], -1)
                negative_output.append(cls_negative_flattened)
                fusion_negative_output.append(fusion_negative_flattened)
                # print(cls_negative_flattened.shape, "cls_negative should be B=20, seq=1*dim=768")

            fusion_negative_output = torch.stack(fusion_negative_output) # (modality, batch, dim) -> 
            fusion_negative_output = (self.fusion_avg_params.unsqueeze(1).unsqueeze(2) * fusion_negative_output).sum(0) # expand into in batch & dim, sum across modality

            data_negative_output = torch.stack(data_negative_output).permute(1, 0, 2) # (modality, batch, dim) -> (batch, modality, dim)
            data_negative_output = data_negative_output.reshape(data_negative_output.shape[0], -1) # (batch, modality, dim) -> (batch, modality*dim)

            negative_output.append(fusion_negative_output) 
            negative_output.append(data_negative_output) #
            negative_output = torch.cat(negative_output, dim=1)
            # print(negative_output.shape, " should be B=20, seq=(3*256)+768")

            
            output_embeddings["z_reparamed_negative"] = negative_output


            # if '4' in str(self.fusion_tokens.device):
            #     print("self.fusion_tokens", self.fusion_tokens[0][0][:7], "grad", self.fusion_tokens[0][0][:7].grad) # [1, ,1, 768]
            #     print("self.cls_anchor", self.cls_anchor['R'][0][0][:7], "grad", self.fusion_tokens[0][0][:7].grad) # [1, ,1, 768]
            # print("self.fusion_tokens", self.fusion_tokens[:5])
            # [ 0.0331,  0.0106,  0.0539,  0.0456,  0.0274,  0.0102, -0.0472],
            
            anchor_output = self.bottleneck(anchor_output)
            output_class = self.decoder(anchor_output)

            return output_class, output_embeddings

        return output_embeddings
