import torch
import torchvision.transforms as TT

from atss_core.config import cfg
from atss_core.data import transforms as T
from atss_core.structures.image_list import to_image_list
from atss_core.structures.bounding_box import BoxList
from atss_core.structures.boxlist_ops import cat_boxlist
from atss_core.layers import nms as _box_nms
import numpy as np


def im_detect_bbox_aug_vote(model, images, device):
    # Collect detections computed under different transformations
    boxlists_ts = []
    for _ in range(len(images)):
        boxlists_ts.append([])

    def add_preds_t(boxlists_t):
        for i, boxlist_t in enumerate(boxlists_t):
            if len(boxlists_ts[i]) == 0:
                # The first one is identity transform, no need to resize the boxlist
                boxlists_ts[i].append(boxlist_t)
            else:
                # Resize the boxlist as the first one
                boxlists_ts[i].append(boxlist_t.resize(boxlists_ts[i][0].size))

    # Compute detections for the original image (identity transform)
    boxlists_i = im_detect_bbox(model, images, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, device)
    add_preds_t(boxlists_i)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        boxlists_hf = im_detect_bbox_hflip(model, images, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, device)
        add_preds_t(boxlists_hf)

    for idx, scale in enumerate(cfg.TEST.BBOX_AUG.SCALES):
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        min_range = cfg.TEST.BBOX_AUG.SCALE_RANGES[idx][0]
        max_range = cfg.TEST.BBOX_AUG.SCALE_RANGES[idx][1]
        if scale < 800:
            max_size = cfg.INPUT.MAX_SIZE_TEST

        boxlists_scl = im_detect_bbox_scale(model, images, scale, max_size, device)
        boxlists_scl = remove_boxes(boxlists_scl, min_range, max_range)
        add_preds_t(boxlists_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            boxlists_scl_hf = im_detect_bbox_scale(model, images, scale, max_size, device, hflip=True)
            boxlists_scl_hf = remove_boxes(boxlists_scl_hf, min_range, max_range)
            add_preds_t(boxlists_scl_hf)

    # Merge boxlists detected by different bbox aug params
    boxlists = []
    for _, boxlist_ts in enumerate(boxlists_ts):
        bbox = torch.cat([boxlist_t.bbox for boxlist_t in boxlist_ts])
        scores = torch.cat(
            [boxlist_t.get_field('scores') for boxlist_t in boxlist_ts])
        labels = torch.cat(
            [boxlist_t.get_field('labels') for boxlist_t in boxlist_ts])
        boxlist = BoxList(bbox, boxlist_ts[0].size, boxlist_ts[0].mode)
        boxlist.add_field('scores', scores)
        boxlist.add_field('labels', labels)
        boxlists.append(boxlist)
    results = merge_result_from_multi_scales(boxlists, cfg.TEST.BBOX_AUG.MERGE_TYPE, cfg.TEST.BBOX_AUG.VOTE_TH)
    return results


def im_detect_bbox(model, images, target_scale, target_max_size, device):
    """
    Performs bbox detection on the original image.
    """
    transform = TT.Compose([
        T.Resize(target_scale, target_max_size),
        TT.ToTensor(),
        T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN,
            std=cfg.INPUT.PIXEL_STD,
            to_bgr255=cfg.INPUT.TO_BGR255)
    ])
    images = [transform(image) for image in images]
    images = to_image_list(images, cfg.DATALOADER.SIZE_DIVISIBILITY)
    return model(images.to(device))


def im_detect_bbox_hflip(model, images, target_scale, target_max_size, device):
    """
    Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    transform = TT.Compose([
        T.Resize(target_scale, target_max_size),
        TT.RandomHorizontalFlip(1.0),
        TT.ToTensor(),
        T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN,
            std=cfg.INPUT.PIXEL_STD,
            to_bgr255=cfg.INPUT.TO_BGR255)
    ])
    images = [transform(image) for image in images]
    images = to_image_list(images, cfg.DATALOADER.SIZE_DIVISIBILITY)
    boxlists = model(images.to(device))

    # Invert the detections computed on the flipped image
    boxlists_inv = [boxlist.transpose(0) for boxlist in boxlists]
    return boxlists_inv


def im_detect_bbox_scale(model, images, target_scale, target_max_size, device, hflip=False):
    """
    Computes bbox detections at the given scale.
    Returns predictions in the scaled image space.
    """
    if hflip:
        boxlists_scl = im_detect_bbox_hflip(model, images, target_scale,
                                            target_max_size, device)
    else:
        boxlists_scl = im_detect_bbox(model, images, target_scale,
                                      target_max_size, device)
    return boxlists_scl


def remove_boxes(boxlist_ts, min_scale, max_scale):
    new_boxlist_ts = []
    for _, boxlist_t in enumerate(boxlist_ts):
        mode = boxlist_t.mode
        boxlist_t = boxlist_t.convert("xyxy")
        boxes = boxlist_t.bbox
        keep = []
        for j, box in enumerate(boxes):
            w = box[2] - box[0] + 1
            h = box[3] - box[1] + 1
            if (w * h > min_scale * min_scale) and (w * h < max_scale * max_scale):
                keep.append(j)
        new_boxlist_ts.append(boxlist_t[keep].convert(mode))
    return new_boxlist_ts


def merge_result_from_multi_scales(boxlists, nms_type='nms', vote_thresh=0.65):
    num_images = len(boxlists)
    results = []
    for i in range(num_images):
        scores = boxlists[i].get_field("scores")
        labels = boxlists[i].get_field("labels")
        boxes = boxlists[i].bbox
        boxlist = boxlists[i]
        result = []
        # skip the background
        for j in range(1, cfg.MODEL.RETINANET.NUM_CLASSES):
            inds = (labels == j).nonzero().view(-1)

            scores_j = scores[inds]
            boxes_j = boxes[inds, :].view(-1, 4)
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(boxlist_for_class, cfg.MODEL.ATSS.NMS_TH, score_field="scores",
                                            nms_type=nms_type, vote_thresh=vote_thresh)
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field("labels", torch.full((num_labels,), j, dtype=torch.int64, device=scores.device))
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > cfg.MODEL.ATSS.PRE_NMS_TOP_N > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(),
                number_of_detections - cfg.MODEL.ATSS.PRE_NMS_TOP_N + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        results.append(result)
    return results


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores", nms_type='nms', vote_thresh=0.65):
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    if nms_type == 'nms':
        keep = _box_nms(boxes, score, nms_thresh)
        if max_proposals > 0:
            keep = keep[: max_proposals]
        boxlist = boxlist[keep]
    else:
        if nms_type == 'vote':
            boxes_vote, scores_vote = bbox_vote(boxes, score, vote_thresh)
        else:
            boxes_vote, scores_vote = soft_bbox_vote(boxes, score, vote_thresh)
        if len(boxes_vote) > 0:
            boxlist.bbox = boxes_vote
            boxlist.extra_fields['scores'] = scores_vote
    return boxlist.convert(mode)


def bbox_vote(boxes, scores, vote_thresh):
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy().reshape(-1, 1)
    det = np.concatenate((boxes, scores), axis=1)
    if det.shape[0] <= 1:
        return np.zeros((0, 5)), np.zeros((0, 1))
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= vote_thresh)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    boxes = torch.from_numpy(dets[:, :4]).float().cuda()
    scores = torch.from_numpy(dets[:, 4]).float().cuda()

    return boxes, scores


def soft_bbox_vote(boxes, scores, vote_thresh):
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy().reshape(-1, 1)
    det = np.concatenate((boxes, scores), axis=1)
    if det.shape[0] <= 1:
        return np.zeros((0, 5)), np.zeros((0, 1))
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= vote_thresh)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= cfg.MODEL.RETINANET.INFERENCE_TH)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((det_accu_sum, soft_det_accu))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]

    boxes = torch.from_numpy(dets[:, :4]).float().cuda()
    scores = torch.from_numpy(dets[:, 4]).float().cuda()

    return boxes, scores
