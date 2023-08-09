from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


class Combined_Loss(nn.Module):
    """A class used to combine various loss functions for model training.

    Args:
        cfgs: Configuration dictionary.
        fabric: Fabric object for distributed training.

    Attributes:
        cfgs: Configuration dictionary.
        fabric: Fabric object for distributed training.
        triplet: Instance of Triplet_Loss class.
        center: Instance of Center_Loss class.
        ce: Cross entropy loss function.
        context: Instance of ContextualSimilarityLoss class.
        triplet2: Instance of Triplet_Loss class.
    """

    def __init__(self, cfgs: Dict[str, Any], fabric: Any):
        super(Combined_Loss, self).__init__()

        self.cfgs = cfgs
        self.fabric = fabric
        self.val_device = torch.device(cfgs.gpus[0])

        if "triplet" in cfgs.loss_fn:
            self.triplet = Triplet_Loss(cfgs=cfgs, fabric=fabric)
        if "center" in cfgs.loss_fn:
            self.center = Center_Loss(cfgs=cfgs, fabric=fabric)
        if "ce" in cfgs.loss_fn:
            self.ce = nn.CrossEntropyLoss(label_smoothing=cfgs.label_smoothing)
        if "context" in cfgs.loss_fn:
            self.context = ContextualSimilarityLoss(cfgs=cfgs, fabric=fabric)
        if "2triplet" in cfgs.loss_fn:
            self.triplet2 = Triplet_Loss(cfgs=cfgs, fabric=fabric)
        if "orthogonal" in cfgs.loss_fn:
            self.orthogonal = OrthogonalLoss(cfgs=cfgs, fabric=fabric)


    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, output: torch.Tensor,
                target: torch.Tensor, neg_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass to compute the combined loss.

        Args:
            anchor: Anchor embeddings.
            pos: Positive embeddings.
            neg: Negative embeddings.
            output: Model output.
            target: Ground truth labels.
            neg_data: Negative data embeddings. Default to None.

        Returns:
            loss: The computed combined loss.
        """
        loss = 0.0

        if "triplet" in self.cfgs.loss_fn:
            triplet_weighting = self.cfgs.loss_weighting
            triplet_loss = self.triplet(anchor, pos, neg)
            loss += (triplet_weighting * triplet_loss)
        if "center" in self.cfgs.loss_fn:
            center_weighting = self.cfgs.center_weighting
            center_loss = self.center(anchor, target)
            loss += (center_weighting * center_loss)
        if "ce" in self.cfgs.loss_fn:
            ce_weighting = self.cfgs.loss_weighting
            ce_loss = self.ce(output, target)
            loss += (ce_weighting * ce_loss)
        if "context" in self.cfgs.loss_fn:
            context_weighting = self.cfgs.loss_weighting
            context_loss = self.context(anchor, pos, neg)
            loss += (context_weighting * context_loss)
        if "2triplet" in self.cfgs.loss_fn and neg_data is not None:
            triplet_weighting2 = self.cfgs.loss_weighting
            triplet_loss2 = self.triplet2(anchor, pos, neg_data)
            loss += (triplet_weighting2 * triplet_loss2)
        if "orthogonal" in self.cfgs.loss_fn:
            ortho_weighting = self.cfgs.ortho_weighting
            ortho = self.orthogonal(anchor)
            loss += (ortho * ortho_weighting)

        loss *= self.cfgs.alpha

        if self.fabric.device == self.val_device:
            if self.cfgs.use_wandb:
                if "triplet" in self.cfgs.loss_fn:
                    wandb.log({"train/loss_triplet": triplet_loss.item()})
                if "center" in self.cfgs.loss_fn:
                    wandb.log({"train/loss_center": center_loss.item()})
                if "ce" in self.cfgs.loss_fn:
                    wandb.log({"train/loss_ce": ce_loss.item()})
                if "2triplet" in self.cfgs.loss_fn and neg_data is not None:
                    wandb.log({"train/loss_triplet2": triplet_loss2.item()})
                if "orthogonal" in self.cfgs.loss_fn:
                    wandb.log({"train/orthogonal": ortho.item()})

            print_str = ""
            if "triplet" in self.cfgs.loss_fn:
                print_str += "triplet: " + str(triplet_loss.item())
            if "center" in self.cfgs.loss_fn:
                print_str += " center: " + str(center_loss.item())
            if "ce" in self.cfgs.loss_fn:
                print_str += " ce-loss: " + str(ce_loss.item())
            if "2triplet" in self.cfgs.loss_fn and neg_data is not None:
                print_str += " triplet2: " + str(triplet_loss2.item())
            print(print_str)

        return loss


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [B, m, d]
      y: pytorch Variable, with shape [B, n, d]
    Returns:
      dist: pytorch Variable, with shape [B, m, n]
    """
    B = x.size(0)
    m, n = x.size(1), y.size(1)
    x_norm = torch.pow(x, 2).sum(2, keepdim=True).sqrt().expand(B, m, n)
    y_norm = torch.pow(y, 2).sum(2, keepdim=True).sqrt().expand(B, n, m).transpose(-2, -1)
    xy_intersection = x @ y.transpose(-2, -1)
    dist = xy_intersection / (x_norm * y_norm)

    return torch.abs(dist)
  
  
class OrthogonalLoss(nn.Module):
    def __init__(self, cfgs: Dict[str, Any], fabric: Any):
        super(OrthogonalLoss, self).__init__()
        self.cfgs = cfgs
        self.fabric = fabric

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        # Extract modality_tokens and avg_fusion_token from anchor
        unflattened_anchor = torch.unflatten(anchor, 1, (-1, self.cfgs.vit_embed_dim))
        modality_tokens = unflattened_anchor[:, :-1]
        #avg_fusion_token = anchor[:, -1]

        B, N, C = modality_tokens.shape
        dist_mat = cosine_dist(modality_tokens, modality_tokens)  # B*N*N

        top_triu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        _dist = dist_mat[:, top_triu]

        if self.cfgs.dynamic_balancer:

          weight = F.softmax(_dist, dim=-1)
          dist = torch.mean(torch.sum(weight*_dist, dim=1))

        else:
          dist = torch.mean(_dist, dim=(0, 1))

        return dist








    

class Triplet_CE_Loss(nn.Module):
    """A class used to compute a combined loss of triplet loss and cross entropy loss.

    Args:
        cfgs: Configuration dictionary.
        fabric: Fabric object for distributed training.

    Attributes:
        ce_loss: Cross entropy loss function.
        sm_loss: Soft margin loss function.
        triplet_loss: Triplet margin loss function.
        cfgs: Configuration dictionary.
        fabric: Fabric object for distributed training.
        weight: The weight factor for combining the losses.
        distance_function: Distance function to use in the loss computation.
    """

    def __init__(self, cfgs: Dict[str, Any], fabric: Any):
        super(Triplet_CE_Loss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=cfgs.label_smoothing)
        if "soft_margin" in cfgs.loss_fn:
            self.sm_loss = nn.SoftMarginLoss()
        if "triplet_euclidean" in cfgs.loss_fn:
            self.triplet_loss = nn.TripletMarginLoss(margin=cfgs.triplet_loss_margin)
            self.distance_function = nn.PairwiseDistance()
        elif "triplet_cosine" in cfgs.loss_fn:
            self.distance_function = nn.CosineSimilarity()
            self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=self.distance_function,
                                                                 margin=cfgs.triplet_loss_margin)
        self.cfgs = cfgs
        self.fabric = fabric
        self.weight = cfgs.loss_weighting

    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
                output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the Triplet_CE_Loss.

        Args:
            anchor: Anchor embeddings.
            pos: Positive embeddings.
            neg: Negative embeddings.
            output: Model output.
            target: Ground truth labels.

        Returns:
            loss: The computed Triplet_CE_Loss.
        """
        clamped_weight = self.weight

        classification_loss = self.ce_loss(output, target)

        if "soft_margin" in self.cfgs.loss_fn:
            ap_distance = self.distance_function(anchor, pos)
            an_distance = self.distance_function(anchor, neg)
            triplet = self.sm_loss(an_distance - ap_distance, torch.ones_like(an_distance))
        else:
            triplet = self.triplet_loss(anchor, pos, neg)

        if self.cfgs.loss_scaling:
            triplet = 5e2 * triplet
            classification_loss = 1e1 * classification_loss
        else:
            triplet = self.cfgs.alpha * (1.0 - clamped_weight) * triplet
            classification_loss = self.cfgs.alpha * clamped_weight * classification_loss

        loss = triplet + classification_loss

        if self.fabric.device == self.val_device:
            if self.cfgs.use_wandb:
                wandb.log({"train/loss_triplet": triplet.item(), "train/loss_ce": classification_loss.item()})
            print("triplet:", triplet.item(), "ce-loss:", classification_loss.item())

        return loss


class Triplet_Loss(nn.Module):
    """A class used to compute the triplet loss.

    Args:
        cfgs: Configuration dictionary.
        fabric: Fabric object for distributed training.

    Attributes:
        sm_loss: Soft margin loss function.
        triplet_loss: Triplet margin loss function.
        cfgs: Configuration dictionary.
        fabric: Fabric object for distributed training.
        distance_function: Distance function to use in the loss computation.
    """

    def __init__(self, cfgs: Dict[str, Any], fabric: Any):
        super(Triplet_Loss, self).__init__()

        if "soft_margin" in cfgs.loss_fn:
            self.sm_loss = nn.SoftMarginLoss()
        if "triplet_euclidean" in cfgs.loss_fn:
            self.triplet_loss = nn.TripletMarginLoss(margin=cfgs.triplet_loss_margin)
            self.distance_function = nn.PairwiseDistance()
        elif "triplet_cosine" in cfgs.loss_fn:
            self.distance_function = nn.CosineSimilarity()
            self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=self.distance_function,
                                                                 margin=cfgs.triplet_loss_margin)
        self.cfgs = cfgs
        self.fabric = fabric

    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the Triplet_Loss.

        Args:
            anchor: Anchor embeddings.
            pos: Positive embeddings.
            neg: Negative embeddings.

        Returns:
            loss: The computed Triplet_Loss.
        """
        if "soft_margin" in self.cfgs.loss_fn:
            ap_distance = self.distance_function(anchor, pos)
            an_distance = self.distance_function(anchor, neg)
            triplet = self.sm_loss(an_distance - ap_distance, torch.ones_like(an_distance))
        else:
            triplet = self.triplet_loss(anchor, pos, neg)

        return triplet


class Center_Loss(nn.Module):
    """A class used to compute the center loss.

    Args:
        cfgs: Configuration dictionary.
        fabric: Fabric object for distributed training.

    Attributes:
        cfgs: Configuration dictionary.
        fabric: Fabric object for distributed training.
        num_classes: Number of classes.
        feat_dim: Feature dimension.
        centers: Centers of the classes.
    """

    def __init__(self, cfgs: Dict[str, Any], fabric: Any):
        super(Center_Loss, self).__init__()

        self.cfgs = cfgs
        self.fabric = fabric
        self.num_classes = cfgs.model_decoder_output_class_num

        if self.cfgs.model_anchor_only_reid:
            self.feat_dim = cfgs.vit_embed_dim * (cfgs.model_num_fusion_tokens)
        else:
            self.feat_dim = cfgs.vit_embed_dim * (
                len(self.cfgs.model_modalities) * cfgs.model_num_cls_tokens + cfgs.model_num_fusion_tokens)

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.fabric.to_device(self.centers)


    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the Center_Loss.

        Args:
            x: Input tensor.
            labels: Ground truth labels.

        Returns:
            loss: The computed Center_Loss.
        """
        batch_size = x.size(0)
        distmat = self.fabric.to_device(torch.pow(x, 2).sum(dim=1, keepdim=True).expand(
            batch_size, self.num_classes)) + \
            self.fabric.to_device(torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                self.num_classes, batch_size)).t()
        self.fabric.to_device(distmat)
        distmat.addmm_(x, self.fabric.to_device(self.centers.t()), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        self.fabric.to_device(classes)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        self.fabric.to_device(labels)
        mask = labels.eq(self.fabric.to_device(classes.expand(batch_size, self.num_classes)))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class Circle_Loss(nn.Module):
    """A class used to compute the circle loss.

    Args:
        cfgs: Configuration dictionary.
        fabric: Fabric object for distributed training.

    Attributes:
        cfgs: Configuration dictionary.
        fabric: Fabric object for distributed training.
        m: Margin value.
        gamma: Gamma value.
        soft_plus: Softplus function.
        cross_entropy: Cross entropy loss function.
    """

    def __init__(self, cfgs: Dict[str, Any], fabric: Any):
        super(Circle_Loss, self).__init__()

        self.cfgs = cfgs
        self.m = self.cfgs.circle_loss_m
        self.gamma = self.cfgs.circle_loss_gamma
        self.fabric = fabric
        self.soft_plus = nn.Softplus()
        self.cross_entropy = nn.CrossEntropyLoss()

    def convert_label_to_similarity(self, normed_feature: torch.Tensor,
                                    label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert label to similarity.

        Args:
            normed_feature: Normalized feature tensor.
            label: Label tensor.

        Returns:
            Tuple of positive and negative similarity tensors.
        """
        similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
        label = torch.cat([label, label], dim=0)
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)

        return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

    def forward(self, feat: torch.Tensor, output: torch.Tensor, lbl: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the Circle_Loss.

        Args:
            feat: Feature tensor.
            output: Output tensor.
            lbl: Label tensor.

        Returns:
            loss: The computed Circle_Loss.
        """
        sp, sn = self.convert_label_to_similarity(feat, lbl)

        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        circle_loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        ce_loss = self.cross_entropy(output, lbl)

        if self.fabric.device == self.val_device:
            if self.cfgs.use_wandb:
                wandb.log({"train/loss_circle": circle_loss.item(), "train/loss_ce": ce_loss.item()})
            print("circle-loss:", circle_loss.item(), "ce-loss:", ce_loss.item())

        return circle_loss + ce_loss


class ContextualSimilarityLoss(nn.Module):
    """A class used to compute the contextual similarity loss.

    Args:
        cfgs: Configuration dictionary.
        fabric: Fabric object for distributed training.
        pos_margin: Positive margin value.
        neg_margin: Negative margin value.
        normalize: Whether to normalize the input tensors.
        eps: Epsilon value to prevent division by zero.

    Attributes:
        cfgs: Configuration dictionary.
        fabric: Fabric object for distributed training.
        pos_margin: Positive margin value.
        neg_margin: Negative margin value.
        normalize: Whether to normalize the input tensors.
        eps: Epsilon value to prevent division by zero.
    """

    def __init__(self, cfgs: Dict[str, Any], fabric: Any, pos_margin: float = 0.75, neg_margin: float = 0.6,
                 normalize: bool = True, eps: float = 0.05):
        super(ContextualSimilarityLoss, self).__init__()
        self.cfgs = cfgs
        self.fabric = fabric
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.normalize = normalize
        self.eps = eps

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the ContextualSimilarityLoss.

        Args:
            anchor: Anchor embeddings.
            positive: Positive embeddings.
            negative: Negative embeddings.

        Returns:
            loss: The computed ContextualSimilarityLoss.
        """
        assert anchor.shape == positive.shape == negative.shape, "All input tensors should have the same shape"

        if self.normalize:
            anchor = F.normalize(anchor, p=2, dim=-1)
            positive = F.normalize(positive, p=2, dim=-1)
            negative = F.normalize(negative, p=2, dim=-1)

        jaccard_pos = self._compute_jaccard(anchor, positive)
        jaccard_neg = self._compute_jaccard(anchor, negative)

        loss_pos = F.relu(jaccard_pos - self.pos_margin).pow(2)
        loss_neg = F.relu(self.neg_margin - jaccard_neg).pow(2)

        loss_pos = loss_pos.mean()
        loss_neg = loss_neg.mean()

        if self.fabric.device == self.val_device:
            if self.cfgs.use_wandb:
                wandb.log({"train/loss_pos": loss_pos.item(), "train/loss_neg": loss_neg.item(),
                           "train/loss_context": loss_pos.item() + loss_neg.item()})
            print("context-loss:", loss_pos.item() + loss_neg.item(), "pos-loss:", loss_pos.item(),
                  "neg-loss:", loss_neg.item())

        return loss_pos + loss_neg

    def _compute_jaccard(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute the Jaccard index.

        Args:
            a: Tensor A.
            b: Tensor B.

        Returns:
            The Jaccard index tensor.
        """
        intersection = (a * b).sum(dim=-1)
        union = a.norm(p=2, dim=-1).pow(2) + b.norm(p=2, dim=-1).pow(2) - intersection
        return intersection / (union.clamp(min=self.eps))
