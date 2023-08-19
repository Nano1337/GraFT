

            # z_pos = {}
            # z_pos_joint = {}
            # fusion_pos_output = []
            # pos_output = []
            # cls_pos = {}
            # for modality in self.cfg.model_modalities:
            
            #     # pass through pretrained encoder
            #     z_pos[modality] = self.transformer(input_positives[modality])

            #     # (B, S, D) -> (S, B, D)
            #     z_pos[modality] = z_pos[modality].last_hidden_state.permute(
            #         1, 0, 2)
                
            #     # create modality tokens
            #     cls_pos[modality] = self.cls_anchor[modality].repeat(
            #         1, z_pos[modality].shape[1], 1)
            #     if self.cfg.model_fusion_combos[1] == "f":
            #         z_pos[modality] = torch.cat(
            #             (cls_pos[modality],
            #              self.fusion_tokens.repeat(1, z_pos[modality].shape[1], 1), z_pos[modality]),
            #             dim=0)
            #     elif self.cfg.model_fusion_combos[1] == "d":
            #         z_pos[modality] = torch.cat(
            #             (cls_pos[modality], z_pos[modality],
            #              self.fusion_tokens.repeat(1, z_pos[modality].shape[1], 1)),
            #             dim=0)

            #     # pass through individual modality transformers
            #     z_pos_joint[modality] = self.modality_transformers[modality](
            #         z_pos[modality])

            #     # (S, B, D) -> (B, S, D)
            #     z_pos_joint[modality] = z_pos_joint[modality].permute(
            #         1, 0, 2)
            #     cls_pos[modality] = cls_pos[modality].permute(1, 0, 2)

            #     # extract fusion tokens
            #     fusion_pos = z_pos_joint[
            #         modality][:, self.cfg.model_num_cls_tokens:self.cfg.model_num_cls_tokens +
            #                   self.cfg.model_num_fusion_tokens, :]
            #     if not self.cfg.lagging_modality_token:
            #         cls_pos[modality] = z_pos_joint[modality][:, :self.cfg.model_num_cls_tokens, :]

            #     # flatten
            #     cls_pos_flattened = cls_pos[modality].reshape(
            #         cls_pos[modality].shape[0], -1)
            #     fusion_pos_flattened = fusion_pos.reshape(
            #         fusion_pos.shape[0], -1)
            #     pos_output.append(cls_pos_flattened)
            #     fusion_pos_output.append(fusion_pos_flattened)
            # fusion_pos_output = torch.stack(fusion_pos_output)
            # fusion_pos_output = (self.fusion_avg_params.unsqueeze(1).unsqueeze(2) * fusion_pos_output).sum(0)

            
            # pos_output.append(fusion_pos_output)
            # pos_output = torch.cat(pos_output, dim=1)

            # z_neg = {}
            # z_neg_joint = {}
            # fusion_neg_output = []
            # neg_output = []
            # cls_neg = {}
            # for modality in self.cfg.model_modalities:

            #     # pass through pretrained encoder
            #     z_neg[modality] = self.transformer(input_negatives[modality])

            #     # (B, S, D) -> (S, B, D)
            #     z_neg[modality] = z_neg[modality].last_hidden_state.permute(
            #         1, 0, 2)

            #     # create modality tokens
            #     cls_neg[modality] = self.cls_anchor[modality].repeat(
            #         1, z_neg[modality].shape[1], 1)

            #     # concat modality, fusion, and data tokens
            #     if self.cfg.model_fusion_combos[2] == "f":
            #         z_neg[modality] = torch.cat(
            #             (cls_neg[modality],
            #              self.fusion_tokens.repeat(1, z_neg[modality].shape[1], 1), z_neg[modality]),
            #             dim=0)
            #     elif self.cfg.model_fusion_combos[2] == "d":
            #         z_neg[modality] = torch.cat(
            #             (cls_neg[modality], z_neg[modality],
            #              self.fusion_tokens.repeat(1, z_neg[modality].shape[1], 1)),
            #             dim=0)

            #     # pass through individual modality transformers
            #     z_neg_joint[modality] = self.modality_transformers[modality](
            #         z_neg[modality])

            #     # (S, B, D) -> (B, S, D)
            #     z_neg_joint[modality] = z_neg_joint[modality].permute(
            #         1, 0, 2)
            #     cls_neg[modality] = cls_neg[modality].permute(1, 0, 2)

            #     # extract fusion tokens
            #     if self.cfg.model_fusion_combos[2] == "f":
            #         fusion_neg = z_neg_joint[
            #             modality][:, self.cfg.model_num_cls_tokens:self.cfg.model_num_cls_tokens +
            #                       self.cfg.model_num_fusion_tokens, :]
            #     else:
            #         fusion_neg = z_neg_joint[
            #             modality][:, self.cfg.model_num_cls_tokens +
            #                       self.cfg.data_token_step:self.cfg.model_num_cls_tokens +
            #                       self.cfg.model_num_fusion_tokens +
            #                       self.cfg.data_token_step, :]
                
            #     if not self.cfg.lagging_modality_token:
            #         cls_neg[modality] = z_neg_joint[modality][:, :self.cfg.model_num_cls_tokens, :]
                
            #     # flatten
            #     fusion_neg_flattened = fusion_neg.reshape(
            #         fusion_neg.shape[0], -1)
            #     cls_neg_flattened = cls_neg[modality].reshape(
            #         cls_neg[modality].shape[0], -1)
            #     neg_output.append(cls_neg_flattened)
            #     fusion_neg_output.append(fusion_neg_flattened)
            # # fusion_neg_output = torch.mean(torch.stack(fusion_neg_output),
            # #                                dim=0)
            # fusion_neg_output = torch.stack(fusion_neg_output)
            # fusion_neg_output = (self.fusion_avg_params.unsqueeze(1).unsqueeze(2) * fusion_neg_output).sum(0)

            # neg_output.append(fusion_neg_output)
            # neg_output = torch.cat(neg_output, dim=1)

            # output_embeddings["z_reparamed_positive"] = pos_output
            # output_embeddings["z_reparamed_negative"] = neg_output

            # self.fusion_avg_params.retain_grad()
            # if '1' in str(self.fusion_avg_params.device):
            #     print("self.fusion_avg_params", self.fusion_avg_params, self.fusion_avg_params.requires_grad, self.fusion_avg_params.grad)
