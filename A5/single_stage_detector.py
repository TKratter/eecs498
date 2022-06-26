import time
import math
import torch
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt


def hello_single_stage_detector():
    print("Hello from single_stage_detector.py!")


def GenerateAnchor(anc, grid):
    """
    Anchor generator.

    Inputs:
    - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
      each point in the grid. anc[a] = (w, h) gives the width and height of the
      a'th anchor shape.
    - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
      center of each feature from the backbone feature map. This is the tensor
      returned from GenerateGrid.

    Outputs:
    - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
      anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
      centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
      boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
      and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
      corners of the box.
    """
    anchors = None
    ##############################################################################
    # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
    # generate all the anchor coordinates for each image. Support batch input.   #
    ##############################################################################
    # Replace "pass" statement with your code
    A, _ = anc.shape
    B, H, W, _ = grid.shape
    anchors = grid.unsqueeze(1).repeat(1, A, 1, 1, 2)
    anchors[:, :, :, :, :2] -= anc.view(1, A, 1, 1, 2) / 2
    anchors[:, :, :, :, 2:] += anc.view(1, A, 1, 1, 2) / 2
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return anchors


def GenerateProposal(anchors, offsets, method='YOLO'):
    """
    Proposal generator.

    Inputs:
    - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
      by the coordinates of their top-left and bottom-right corners.
    - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
      convert anchor boxes into region proposals. The transformation
      offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
      anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
      (-0.5, 0.5).
    - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'

    Outputs:
    - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
      coordinates of their top-left and bottom-right corners. Applying the
      transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
      proposal proposals[b, a, h, w].

    """
    assert (method in ['YOLO', 'FasterRCNN'])
    proposals = None
    ##############################################################################
    # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
    # compute the proposal coordinates using the transformation formulas above.  #
    ##############################################################################
    # Replace "pass" statement with your code
    B, A, H, W, _ = anchors.shape
    proposals = torch.empty_like(anchors)
    ha = (anchors[:, :, :, :, 1:2] - anchors[:, :, :, :, 3:4]).abs()
    wa = (anchors[:, :, :, :, 0:1] - anchors[:, :, :, :, 2:3]).abs()
    ya = (anchors[:, :, :, :, 1:2] + anchors[:, :, :, :, 3:4]) / 2
    xa = (anchors[:, :, :, :, 0:1] + anchors[:, :, :, :, 2:3]) / 2

    tx, ty, tw, th = torch.split(offsets, dim=-1, split_size_or_sections=1)

    if method == "YOLO":
        centers = torch.cat([xa, ya], dim=-1) + torch.cat([tx, ty], dim=-1)
    else:
        centers = torch.cat([xa, ya], dim=-1) + torch.cat([tx, ty], dim=-1) * torch.cat([wa, ha], dim=-1)

    sizes = torch.cat([wa, ha], dim=-1) * torch.exp(torch.cat([tw, th], dim=-1))

    proposals[:, :, :, :, :2] = centers - sizes / 2
    proposals[:, :, :, :, 2:] = centers + sizes / 2
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return proposals


def IoU(proposals, bboxes):
    """
    Compute intersection over union between sets of bounding boxes.

    Inputs:
    - proposals: Proposals of shape (B, A, H', W', 4)
    - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
      Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
      If image i has fewer than N boxes, then bboxes[i] will be padded with extra
      rows of -1.

    Outputs:
    - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
      the IoU between one element of proposals[b] and bboxes[b, n].

    For this implementation you DO NOT need to filter invalid proposals or boxes;
    in particular you don't need any special handling for bboxxes that are padded
    with -1.
    """
    iou_mat = None
    ##############################################################################
    # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
    # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
    # However, you need to make sure to compute the IoU correctly (it should be  #
    # 0 in those cases.                                                          #
    # You need to ensure your implementation is efficient (no for loops).        #
    # HINT:                                                                      #
    # IoU = Area of Intersection / Area of Union, where                          #
    # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
    # and the Area of Intersection can be computed using the top-left corner and #
    # bottom-right corner of proposal and bbox. Think about their relationships. #
    ##############################################################################
    # Replace "pass" statement with your code
    B, A, H, W, _ = proposals.shape
    _, N, _ = bboxes.shape
    iou_mat = torch.zeros(B, A * H * W, N, device=proposals.device, dtype=proposals.dtype)
    px_min, py_min, px_max, py_max = torch.split(proposals.view(B, A * H * W, 4), dim=-1, split_size_or_sections=1)
    bx_min, by_min, bx_max, by_max = [b.permute(0, 2, 1) for b in
                                      torch.split(bboxes[:, :, :4], dim=-1, split_size_or_sections=1)]
    zero_tensor = torch.zeros(1, device=proposals.device, dtype=proposals.dtype)
    intersect_x = torch.maximum(
        torch.maximum(bx_max - px_min, zero_tensor) - torch.maximum(bx_max - px_max, zero_tensor) - torch.maximum(
            bx_min - px_min, zero_tensor), zero_tensor)
    intersect_y = torch.maximum(
        torch.maximum(by_max - py_min, zero_tensor) - torch.maximum(by_max - py_max, zero_tensor) - torch.maximum(
            by_min - py_min, zero_tensor), zero_tensor)

    intesection = intersect_y * intersect_x
    union = (px_max - px_min) * (py_max - py_min) + (bx_max - bx_min) * (by_max - by_min) - intesection

    iou_mat = intesection / union
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return iou_mat


class PredictionNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
        super().__init__()

        assert (num_classes != 0 and num_anchors != 0)
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        ##############################################################################
        # TODO: Set up a network that will predict outputs for all anchors. This     #
        # network should have a 1x1 convolution with hidden_dim filters, followed    #
        # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       #
        # finally another 1x1 convolution layer to predict all outputs. You can      #
        # use an nn.Sequential for this network, and store it in a member variable.  #
        # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                #
        # A=self.num_anchors and C=self.num_classes.                                 #
        ##############################################################################
        # Make sure to name your prediction network pred_layer.
        self.pred_layer = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=(1, 1)),
                                        nn.Dropout(p=drop_ratio),
                                        nn.LeakyReLU(),
                                        nn.Conv2d(in_channels=hidden_dim,
                                                  out_channels=5 * self.num_anchors + self.num_classes,
                                                  kernel_size=(1, 1)))
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def _extract_anchor_data(self, anchor_data, anchor_idx):
        """
        Inputs:
        - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
          D for each of A anchors at each point in an H x W grid.
        - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

        Returns:
        - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
          of the anchors specified by anchor_idx.
        """
        B, A, D, H, W = anchor_data.shape
        anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
        extracted_anchors = anchor_data[anchor_idx]
        return extracted_anchors

    def _extract_class_scores(self, all_scores, anchor_idx):
        """
        Inputs:
        - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
          C classes at each point in an H x W grid.
        - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
          which to extract classification scores

        Returns:
        - extracted_scores: Tensor of shape (M, C) giving the classification scores
          for each of the anchors specified by anchor_idx.
        """
        B, C, H, W = all_scores.shape
        A = self.num_anchors
        all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
        all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
        all_scores = all_scores.reshape(B * A * H * W, C)
        extracted_scores = all_scores[anchor_idx]
        return extracted_scores

    def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
        """
        Run the forward pass of the network to predict outputs given features
        from the backbone network.

        Inputs:
        - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
          by the backbone network.
        - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
          marked as positive. These are only given during training; at test-time
          this should be None.
        - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
          marked as negative. These are only given at training; at test-time this
          should be None.

        The outputs from this method are different during training and inference.

        During training, pos_anchor_idx and neg_anchor_idx are given and identify
        which anchors should be positive and negative, and this forward pass needs
        to extract only the predictions for the positive and negative anchors.

        During inference, only features are provided and this method needs to return
        predictions for all anchors.

        Outputs (During training):
        - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
          scores for positive anchors and negative anchors (in that order).
        - offsets: Tensor of shape (M, 4) giving predicted transformation for
          positive anchors.
        - class_scores: Tensor of shape (M, C) giving classification scores for
          positive anchors.

        Outputs (During inference):
        - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
          scores for all anchors.
        - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
          all all anchors.
        - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
          each spatial position.
        """
        conf_scores, offsets, class_scores = None, None, None
        ############################################################################
        # TODO: Use backbone features to predict conf_scores, offsets, and         #
        # class_scores. Make sure conf_scores is between 0 and 1 by squashing the  #
        # network output with a sigmoid. Also make sure the first two elements t^x #
        # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  #
        # and subtracting 0.5.                                                     #
        #                                                                          #
        # During training you need to extract the outputs for only the positive    #
        # and negative anchors as specified above.                                 #
        #                                                                          #
        # HINT: You can use the provided helper methods self._extract_anchor_data  #
        # and self._extract_class_scores to extract information for positive and   #
        # negative anchors specified by pos_anchor_idx and neg_anchor_idx.         #
        ############################################################################
        # Replace "pass" statement with your code

        # inference
        all_preds = self.pred_layer(features)  # (B, 5*A+C, 7, 7)
        class_scores = all_preds[:, -self.num_classes:, :, :]
        conf_scores = torch.sigmoid(all_preds[:, 0:5 * self.num_anchors: 5, :, :])
        tx = torch.sigmoid(all_preds[:, 1:5 * self.num_anchors: 5, :, :]).unsqueeze(2) - 0.5
        ty = torch.sigmoid(all_preds[:, 2:5 * self.num_anchors: 5, :, :]).unsqueeze(2) - 0.5
        tw = all_preds[:, 3:5 * self.num_anchors: 5, :, :].unsqueeze(2)
        th = all_preds[:, 4:5 * self.num_anchors: 5, :, :].unsqueeze(2)
        offsets = torch.cat([tx, ty, tw, th], dim=2)

        if not (pos_anchor_idx is None) and not (neg_anchor_idx is None):
            class_scores = self._extract_class_scores(class_scores, pos_anchor_idx)
            offsets = self._extract_anchor_data(offsets, pos_anchor_idx)
            conf_scores = torch.cat([self._extract_anchor_data(conf_scores.unsqueeze(2), pos_anchor_idx),
                                     self._extract_anchor_data(conf_scores.unsqueeze(2), neg_anchor_idx)], dim=0)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return conf_scores, offsets, class_scores


class SingleStageDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.anchor_list = torch.tensor(
            [[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]], device='cuda')  # READ ONLY
        self.feat_extractor = FeatureExtractor()
        self.num_classes = 20
        self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                              num_classes=self.num_classes)

    def forward(self, images, bboxes):
        """
        Training-time forward pass for the single-stage detector.

        Inputs:
        - images: Input images, of shape (B, 3, 224, 224)
        - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

        Outputs:
        - total_loss: Torch scalar giving the total loss for the batch.
        """
        # weights to multiple to each loss term
        w_conf = 1  # for conf_scores
        w_reg = 1  # for offsets
        w_cls = 1  # for class_prob

        total_loss = None
        ##############################################################################
        # TODO: Implement the forward pass of SingleStageDetector.                   #
        # A few key steps are outlined as follows:                                   #
        # i) Image feature extraction,                                               #
        # ii) Grid and anchor generation,                                            #
        # iii) Compute IoU between anchors and GT boxes and then determine activated/#
        #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
        # iv) Compute conf_scores, offsets, class_prob through the prediction network#
        # v) Compute the total_loss which is formulated as:                          #
        #    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,  #
        #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
        #    BboxRegression, and w_cls by ObjectClassification.                      #
        # HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedAnchors in this notebook #
        #       (A5-1) for a better performance than with the default value.         #
        ##############################################################################
        # Replace "pass" statement with your code
        features = self.feat_extractor(images)
        _, offsets, _ = self.pred_network.forward(features)
        grid = GenerateGrid(features.shape[0], dtype=features.dtype, device=features.device)
        anchors = GenerateAnchor(self.anchor_list, grid)
        proposals = GenerateProposal(anchors=anchors, offsets=offsets.permute(0, 1, 3, 4, 2))
        iou_mat = IoU(proposals=proposals, bboxes=bboxes)
        pos_anchor_idx, neg_anchor_idx, GT_conf_scores, GT_offsets, GT_class, \
        activated_anc_coord, negative_anc_coord = ReferenceOnActivatedAnchors(
            anchors=anchors, bboxes=bboxes, grid=grid, iou_mat=iou_mat, neg_thresh=0.2)
        conf_scores, offsets, class_scores = self.pred_network.forward(features=features,
                                                                       pos_anchor_idx=pos_anchor_idx,
                                                                       neg_anchor_idx=neg_anchor_idx)

        conf_loss = ConfScoreRegression(conf_scores=conf_scores, GT_conf_scores=GT_conf_scores)
        reg_loss = BboxRegression(offsets=offsets, GT_offsets=GT_offsets)
        cls_loss = ObjectClassification(class_prob=class_scores, GT_class=GT_class, batch_size=features.shape[0],
                                        anc_per_img=torch.prod(torch.tensor(anchors.shape[1:-1])),
                                        activated_anc_ind=pos_anchor_idx)

        total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        return total_loss

    def inference(self, images, thresh=0.5, nms_thresh=0.7):
        """"
        Inference-time forward pass for the single stage detector.

        Inputs:
        - images: Input images
        - thresh: Threshold value on confidence scores
        - nms_thresh: Threshold value on NMS

        Outputs:
        - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                          a list of B (*x4) tensors
        - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
        - final_class: Corresponding class predictions, a list of B  (*x1) tensors
        """
        final_proposals, final_conf_scores, final_class = [], [], []
        ##############################################################################
        # TODO: Predicting the final proposal coordinates `final_proposals`,         #
        # confidence scores `final_conf_scores`, and the class index `final_class`.  #
        # The overall steps are similar to the forward pass but now you do not need  #
        # to decide the activated nor negative anchors.                              #
        # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
        # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
        # threshold `nms_thresh`.                                                    #
        # The class index is determined by the class with the maximal probability.   #
        # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
        # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
        ##############################################################################
        # Replace "pass" statement with your code
        features = self.feat_extractor(images)
        grid = GenerateGrid(features.shape[0], dtype=features.dtype, device=features.device)
        anchors = GenerateAnchor(self.anchor_list, grid)
        conf_scores, offsets, class_scores = self.pred_network.forward(features)

        anchors = GenerateAnchor(self.anchor_list, grid)
        proposals = GenerateProposal(anchors=anchors, offsets=offsets.permute(0, 1, 3, 4, 2))
        B, A, H, W, _ = proposals.shape

        proposals = proposals.view(B, -1, 4)
        conf_scores = conf_scores.view(B, -1)
        for b in range(B):
            keep_indices = torch.ops.torchvision.nms(proposals[b], conf_scores[b], nms_thresh)
            thresh_indices = torch.argwhere(conf_scores[b, keep_indices] > thresh)
            final_proposals.append(proposals[b, keep_indices, :].detach()[thresh_indices].squeeze(1))
            final_conf_scores.append(
                conf_scores[b, keep_indices].detach()[thresh_indices])
            final_class.append(
                self.pred_network._extract_class_scores(class_scores,
                                                        keep_indices).squeeze().argmax(dim=-1, keepdims=True).detach()[
                    thresh_indices].squeeze(-1))
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return final_proposals, final_conf_scores, final_class


def _iou(b1: torch.Tensor, b2: torch.Tensor) -> float:
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2

    zero_tensor = torch.zeros(1, device=b1.device)

    intersect_x = torch.maximum(
        torch.maximum(x2_max - x1_min, zero_tensor) - torch.maximum(x2_max - x1_max, zero_tensor) - torch.maximum(
            x2_min - x1_min, zero_tensor), zero_tensor)
    intersect_y = torch.maximum(
        torch.maximum(y2_max - y1_min, zero_tensor) - torch.maximum(y2_max - y1_max, zero_tensor) - torch.maximum(
            y2_min - y1_min, zero_tensor), zero_tensor)

    intesection = intersect_y * intersect_x
    union = (x1_max - x1_min) * (y1_max - y1_min) + (x1_max - x1_min) * (y1_max - y1_min) - intesection

    return intesection / union


def nms(boxes, scores, iou_threshold=0.5, topk=None):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Inputs:
    - boxes: top-left and bottom-right coordinate values of the bounding boxes
      to perform NMS on, of shape Nx4
    - scores: scores for each one of the boxes, of shape N
    - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
    - topk: If this is not None, then return only the topk highest-scoring boxes.
      Otherwise if this is None, then return all boxes that pass NMS.

    Outputs:
    - keep: torch.long tensor with the indices of the elements that have been
      kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    #   github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/nms_cpu.cpp  #
    #############################################################################
    # Replace "pass" statement with your code

    keep_indices = sorted(list(range(scores.shape[0])), key=lambda i: scores[i], reverse=True)

    check_indices = keep_indices.copy()

    while len(check_indices) > 0:
        top_score_ind = check_indices[0]

        check_indices.remove(top_score_ind)

        top_score_box = boxes[top_score_ind]

        for i in check_indices:
            if _iou(top_score_box, boxes[i]) >= iou_threshold:
                if i == 2895:
                    print(_iou(top_score_box, boxes[i]))
                keep_indices.remove(i)
                check_indices.remove(i)
    print('done')
    keep = torch.tensor(keep_indices, device=boxes.device)
    if not (topk is None):
        keep = keep[:topk]
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def ConfScoreRegression(conf_scores, GT_conf_scores):
    """
    Use sum-squared error as in YOLO

    Inputs:
    - conf_scores: Predicted confidence scores
    - GT_conf_scores: GT confidence scores

    Outputs:
    - conf_score_loss
    """
    # the target conf_scores for negative samples are zeros
    GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                                torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
    conf_score_loss = torch.sum((conf_scores - GT_conf_scores) ** 2) * 1. / GT_conf_scores.shape[0]
    return conf_score_loss


def BboxRegression(offsets, GT_offsets):
    """"
    Use sum-squared error as in YOLO
    For both xy and wh

    Inputs:
    - offsets: Predicted box offsets
    - GT_offsets: GT box offsets

    Outputs:
    - bbox_reg_loss
    """
    bbox_reg_loss = torch.sum((offsets - GT_offsets) ** 2) * 1. / GT_offsets.shape[0]
    return bbox_reg_loss
