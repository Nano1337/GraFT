import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


'''
Sums triplet, center, and cross entropy loss combinations
* e.g. triplet_euclidean_soft_margin+center+ce
* e.g. triplet_cosine+center+ce
* e.g. triplet_euclidean+ce
'''
class Combined_Loss(nn.Module):
    def __init__(self, cfgs, fabric):
        super(Combined_Loss, self).__init__()

        self.cfgs = cfgs
        self.fabric = fabric

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
        

    def forward(self, anchor, pos, neg, output, target, neg_data=None):
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
            # if neg_data is None:
            #     raise ValueError("neg_data must be provided for 2triplet loss")
            triplet_weighting2 = self.cfgs.loss_weighting
            triplet_loss2 = self.triplet2(anchor, pos, neg_data)
            loss += (triplet_weighting2 * triplet_loss2)

        loss *= self.cfgs.alpha

        # only print on global rank 0
        if self.fabric.is_global_zero:
            if self.cfgs.use_wandb:
                if "triplet" in self.cfgs.loss_fn:
                    wandb.log({"train/loss_triplet": triplet_loss.item()})
                if "center" in self.cfgs.loss_fn:
                    wandb.log({"train/loss_center": center_loss.item()})
                if "ce" in self.cfgs.loss_fn:
                    wandb.log({"train/loss_ce": ce_loss.item()})
                if "2triplet" in self.cfgs.loss_fn and neg_data is not None:
                    wandb.log({"train/loss_triplet2": triplet_loss2.item()})
                
            print_str = ""
            if "triplet" in self.cfgs.loss_fn:
                print_str += "triplet: "+ str(triplet_loss.item())
            if "center" in self.cfgs.loss_fn:
                print_str += " center: " + str(center_loss.item())
            if "ce" in self.cfgs.loss_fn:
                print_str += " ce-loss: " + str(ce_loss.item())
            if "2triplet" in self.cfgs.loss_fn and neg_data is not None:
                print_str += " triplet2: " + str(triplet_loss2.item())
            print(print_str)

        return loss

class Triplet_CE_Loss(nn.Module):
    """
    ReIDLoss is a custom loss function for re-identification tasks. It combines cross entropy loss and triplet margin loss. 
    Depending on the configuration, the weight of the loss combination can be a learnable parameter or a fixed value.

    Attributes:
        ce_loss (nn.Module): Cross entropy loss function.
        triplet_loss (nn.Module): Triplet margin loss function.
        cfgs (dict): Configuration parameters for the loss function.
        fabric: An object which holds methods for distributed training.
        weight (torch.nn.Parameter or float): The weight factor for combining the losses. 
    """

    def __init__(self, cfgs: dict, fabric):
        """
        Initialize ReIDLoss instance.

        Args:
            cfgs (dict): Configuration parameters for the loss function.
            fabric: An object which holds methods for distributed training.
        """
        super(Triplet_CE_Loss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=cfgs.label_smoothing)
        if "soft_margin" in cfgs.loss_fn:
            self.sm_loss = nn.SoftMarginLoss()
        if "triplet_euclidean" in cfgs.loss_fn:
            self.triplet_loss = nn.TripletMarginLoss(margin=cfgs.triplet_loss_margin)
            self.distance_function = nn.PairwiseDistance()
        elif "triplet_cosine" in cfgs.loss_fn:
            self.distance_function = nn.CosineSimilarity()
            self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=self.distance_function, margin=cfgs.triplet_loss_margin)
        self.cfgs = cfgs
        self.fabric = fabric

        # optionally have a learnable loss weighting for the two losses
        self.weight = cfgs.loss_weighting

    def forward(self, anchor, pos, neg, output, target):
        """
        Forward pass to compute the ReIDLoss. 

        Args:
            anchor (torch.Tensor): Anchor embeddings.
            pos (torch.Tensor): Positive embeddings.
            neg (torch.Tensor): Negative embeddings.
            output (torch.Tensor): Model output.
            target (torch.Tensor): Ground truth labels.

        Returns:
            loss (torch.Tensor): The computed ReIDLoss.
        """
        clamped_weight = self.weight

        classification_loss = self.ce_loss(output, target)

        if "soft_margin" in self.cfgs.loss_fn:

            # compute the distance between anchor and positive and anchor and negative
            ap_distance = self.distance_function(anchor, pos)
            an_distance = self.distance_function(anchor, neg)
            
            triplet = self.sm_loss(an_distance - ap_distance, torch.ones_like(an_distance))
        
        else:
            triplet = self.triplet_loss(anchor, pos, neg)

        if self.cfgs.loss_scaling: # false by default
            # print("Scaling losses by 1e7 and 1e1")
            triplet = 5e2 * triplet
            classification_loss = 1e1 * classification_loss
        else:
            triplet = self.cfgs.alpha * (1.0 - clamped_weight) * triplet
            classification_loss = self.cfgs.alpha * clamped_weight * classification_loss

        loss = triplet + classification_loss

        # only print on global rank 0
        if self.fabric.is_global_zero:

            if self.cfgs.use_wandb:
                wandb.log({"train/loss_triplet": triplet.item(), "train/loss_ce": classification_loss.item()})

            print("triplet:", triplet.item(), "ce-loss:", classification_loss.item())

                
        return loss

class Triplet_Loss(nn.Module):
    """
    ReIDLoss is a custom loss function for re-identification tasks. It combines cross entropy loss and triplet margin loss. 
    Depending on the configuration, the weight of the loss combination can be a learnable parameter or a fixed value.

    Attributes:
        ce_loss (nn.Module): Cross entropy loss function.
        triplet_loss (nn.Module): Triplet margin loss function.
        cfgs (dict): Configuration parameters for the loss function.
        fabric: An object which holds methods for distributed training.
        weight (torch.nn.Parameter or float): The weight factor for combining the losses. 
    """

    def __init__(self, cfgs: dict, fabric):
        """
        Initialize ReIDLoss instance.

        Args:
            cfgs (dict): Configuration parameters for the loss function.
            fabric: An object which holds methods for distributed training.
        """
        super(Triplet_Loss, self).__init__()

        if "soft_margin" in cfgs.loss_fn:
            self.sm_loss = nn.SoftMarginLoss()
        if "triplet_euclidean" in cfgs.loss_fn:
            self.triplet_loss = nn.TripletMarginLoss(margin=cfgs.triplet_loss_margin)
            self.distance_function = nn.PairwiseDistance()
        elif "triplet_cosine" in cfgs.loss_fn:
            self.distance_function = nn.CosineSimilarity()
            self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=self.distance_function, margin=cfgs.triplet_loss_margin)
        self.cfgs = cfgs
        self.fabric = fabric


    def forward(self, anchor, pos, neg):
        """
        Forward pass to compute the ReIDLoss. 

        Args:
            anchor (torch.Tensor): Anchor embeddings.
            pos (torch.Tensor): Positive embeddings.
            neg (torch.Tensor): Negative embeddings.

        Returns:
            loss (torch.Tensor): The computed ReIDLoss.
        """

        if "soft_margin" in self.cfgs.loss_fn:

            # compute the distance between anchor and positive and anchor and negative
            ap_distance = self.distance_function(anchor, pos)
            an_distance = self.distance_function(anchor, neg)
            
            triplet = self.sm_loss(an_distance - ap_distance, torch.ones_like(an_distance))
        
        else:
            triplet = self.triplet_loss(anchor, pos, neg)

        return triplet


class Center_Loss(nn.Module):
    def __init__(self, cfgs, fabric):
        super(Center_Loss, self).__init__()

        self.cfgs = cfgs
        self.fabric = fabric
        self.num_classes = cfgs.model_decoder_output_class_num

        if self.cfgs.model_anchor_only_reid:
            self.feat_dim = cfgs.vit_embed_dim * (cfgs.model_num_fusion_tokens)
        else:   
            self.feat_dim = cfgs.vit_embed_dim * (len(self.cfgs.model_modalities) * cfgs.model_num_cls_tokens + cfgs.model_num_fusion_tokens)

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.fabric.to_device(self.centers)

    def forward(self, x, labels):
        '''
        Reference: Bag of Tricks and A Strong Baseline for Deep Person Re-identification, Luo et. al. 
        '''

        batch_size = x.size(0)
        distmat = self.fabric.to_device(torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)) + \
                  self.fabric.to_device(torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size)).t()
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
    def __init__(self, cfgs, fabric) -> None:
        super(Circle_Loss, self).__init__()
        
        self.cfgs = cfgs
        self.m = self.cfgs.circle_loss_m
        self.gamma = self.cfgs.circle_loss_gamma
        self.fabric = fabric
        self.soft_plus = nn.Softplus()
        self.cross_entropy = nn.CrossEntropyLoss()

    def convert_label_to_similarity(self, normed_feature, label):
        similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
        label = torch.cat([label, label], dim=0)
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)

        # print("similarity_matrix shape:", similarity_matrix.shape, "positive_matrix shape:", positive_matrix.shape, "negative_matrix shape:", negative_matrix.shape)
        return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

    def forward(self, feat, output, lbl):
        sp, sn = self.convert_label_to_similarity(feat, lbl)
        

        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        circle_loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        ce_loss = self.cross_entropy(output, lbl)

        # Print everything
        # print("sp:", sp)
        # print("sn:", sn)
        # print("ap:", ap)
        # print("an:", an)
        # print("delta_p:", delta_p)
        # print("delta_n:", delta_n)
        # print("logit_p:", logit_p)
        # print("logit_n:", logit_n)

        # only print on global rank 0
        if self.fabric.is_global_zero:
            if self.cfgs.use_wandb:
                wandb.log({"train/loss_circle": circle_loss.item(), "train/loss_ce": ce_loss.item()})

            print("circle-loss:", circle_loss.item(), "ce-loss:", ce_loss.item())

        return circle_loss + ce_loss
    
class ContextualSimilarityLoss(torch.nn.Module):
    def __init__(self, cfgs, fabric, pos_margin=0.75, neg_margin=0.6, normalize=True, eps=0.05):
        super(ContextualSimilarityLoss, self).__init__()
        self.cfgs = cfgs
        self.fabric = fabric
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.normalize = normalize
        self.eps = eps
    
    def forward(self, anchor, positive, negative):
        assert anchor.shape == positive.shape == negative.shape, "All input tensors should have the same shape"
        
        if self.normalize:
            anchor = F.normalize(anchor, p=2, dim=-1)
            positive = F.normalize(positive, p=2, dim=-1)
            negative = F.normalize(negative, p=2, dim=-1)

        # Compute pairwise Jaccard similarities
        jaccard_pos = self._compute_jaccard(anchor, positive)
        jaccard_neg = self._compute_jaccard(anchor, negative)
        
        # Compute contrastive Jaccard loss
        loss_pos = F.relu(jaccard_pos - self.pos_margin).pow(2)
        loss_neg = F.relu(self.neg_margin - jaccard_neg).pow(2)

        # Average over all loss elements
        loss_pos = loss_pos.mean()
        loss_neg = loss_neg.mean()

        # only print on global rank 0
        if self.fabric.is_global_zero:
            if self.cfgs.use_wandb:
                wandb.log({"train/loss_pos": loss_pos.item(), "train/loss_neg": loss_neg.item(), "train/loss_context": loss_pos.item() + loss_neg.item()})
            print("context-loss:", loss_pos.item() + loss_neg.item(), "pos-loss:", loss_pos.item(), "neg-loss:", loss_neg.item())
        
        return loss_pos + loss_neg

    def _compute_jaccard(self, a, b):
        intersection = (a * b).sum(dim=-1)
        union = a.norm(p=2, dim=-1).pow(2) + b.norm(p=2, dim=-1).pow(2) - intersection
        return intersection / (union.clamp(min=self.eps))
