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
            triplet_weighting = self.cfgs.triplet_loss_weighting
            triplet_loss = self.triplet(anchor, pos, neg)
            loss += (triplet_weighting * triplet_loss)
        if "center" in self.cfgs.loss_fn:
            center_weighting = self.cfgs.center_loss_weighting
            center_loss = self.center(anchor, target)
            loss += (center_weighting * center_loss)
        if "ce" in self.cfgs.loss_fn:
            ce_weighting = self.cfgs.ce_loss_weighting
            ce_loss = self.ce(output, target)
            loss += (ce_weighting * ce_loss)

        if self.fabric.device == self.val_device:
            if self.cfgs.use_wandb:
                if "triplet" in self.cfgs.loss_fn:
                    wandb.log({"train/loss_triplet": triplet_loss.item()})
                if "center" in self.cfgs.loss_fn:
                    wandb.log({"train/loss_center": center_loss.item()})
                if "ce" in self.cfgs.loss_fn:
                    wandb.log({"train/loss_ce": ce_loss.item()})
        

            print_str = ""
            if "triplet" in self.cfgs.loss_fn:
                print_str += "triplet: " + str(triplet_loss.item())
            if "center" in self.cfgs.loss_fn:
                print_str += " center: " + str(center_loss.item())
            if "ce" in self.cfgs.loss_fn:
                print_str += " ce-loss: " + str(ce_loss.item())
            print(print_str)

        return loss

def cosine_dist(x, y):
    """
    Args:
      x: torch Variable, with shape [B, m, d]
      y: torch Variable, with shape [B, n, d]
    Returns:
      dist: torch Variable, with shape [B, m, n]
    """
    B = x.size(0)
    m, n = x.size(1), y.size(1)
    x_norm = torch.pow(x, 2).sum(2, keepdim=True).sqrt().expand(B, m, n)
    y_norm = torch.pow(y, 2).sum(2, keepdim=True).sqrt().expand(B, n, m).transpose(-2, -1)
    xy_intersection = x @ y.transpose(-2, -1)
    dist = xy_intersection / (x_norm * y_norm)

    return torch.abs(dist)
    
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

