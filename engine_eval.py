"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
from sklearn.metrics import r2_score
import numpy as np
import cv2

import torch
import torchvision.transforms as standard_transforms
import torch.nn.functional as F

import util.misc as utils
import util.metrics as metrics
from util.misc import NestedTensor


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def visualization(samples, targets, pred, queries, vis_dir, gt_cnt, split_map=None):
    """                         # pred point & query
    Visualize predictions
    """
    vis_dir_ac = os.path.join(vis_dir, 'gt_ac')
    vis_dir_nac = os.path.join(vis_dir, 'gt_nac')
    os.makedirs(vis_dir_ac, exist_ok=True)
    os.makedirs(vis_dir_nac, exist_ok=True)
    
    gts = [t["points"].tolist() for t in targets]                   # gt point         
    
    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose(
        [
            DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            standard_transforms.ToPILImage(),
        ]
    )

    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert("RGB")).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        h, w = sample_vis.shape[:2]
        # sample_vis = cv2.resize(sample_vis, (512, 512))
        
        # draw ground-truth points (red)
        size = 3
        for i, t in enumerate(gts[idx]):
            # print('gt',(int(t[1]), int(t[0])))
            sample_vis = cv2.circle(
                sample_vis, (int(t[1]), int(t[0])), size + 2, (0, 0, 255), -1
            )
            # cv2.putText(sample_vis, str(i+1), (int(t[1]+3), int(t[0])+3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # draw predictions (green)
        for i, p in enumerate(pred[idx]):
            # print('pred',(int(p[1]), int(p[0])))
            sample_vis = cv2.circle(
                sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1
                )
            # cv2.putText(sample_vis, str(i+1), (int(p[1]+3), int(p[0])+3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # draw point-query
        for i, q in enumerate(queries[idx]):
            sample_vis = cv2.circle(
                sample_vis, (int(q[1]*w), int(q[0]*h)), size, (0, 255, 255), -1
                )
            
            # draw line between query and pred
            q_x, q_y = int(q[1]*w), int(q[0]*h)
            p_x, p_y = int(pred[idx][i][1]), int(pred[idx][i][0])
            overlay = sample_vis.copy()
            cv2.line(overlay, (p_x, p_y), (q_x, q_y), (0, 255, 0), 2) 
            alpha = 0.5  
            sample_vis = cv2.addWeighted(overlay, alpha, sample_vis, 1 - alpha, 0)
            # cv2.putText(sample_vis, str(i+1), (int(q[1]*w-3), int(q[0]*h)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # draw split map
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.3 + sample_vis

        # save image
        if vis_dir is not None:
            # eliminate invalid area
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[: valid_h + 1, : valid_w + 1]

            name = targets[idx]["image_path"].split("/")[-1].split(".")[0]
            
            if gt_cnt > 100:
                cv2.imwrite(
                    os.path.join(
                        vis_dir_ac,
                        "{}_gt{}_pred{}.jpg".format(name, len(gts[idx]), len(pred[idx])),
                    ),
                    sample_vis,
                )
            else:
                cv2.imwrite(
                    os.path.join(
                        vis_dir_nac,
                        "{}_gt{}_pred{}.jpg".format(name, len(gts[idx]), len(pred[idx])),
                    ),
                    sample_vis,
                )


# evaluation
@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    epoch=0,
    vis_dir=None,
    distributed=True,
    criterion=None,
    args=None,
):
    model.eval()
    gt_determined = int(args.gt_determined) if args else 100
    metric_logger = utils.MetricLogger(delimiter="  ", win_size=len(data_loader))
    header = "Test:"

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
        vis_dir_ac = os.path.join(vis_dir, 'gt_ac')
        os.makedirs(vis_dir_ac, exist_ok=True)

    gt_cnt_all, pd_cnt_all, gt_cnt_all_ac, pd_cnt_all_ac = [], [], [], []
    print_freq = 10
    
    count = 0
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        # name = targets[0]['image_path'].split("/")[-1].split(".")[0]
        # print(name, end=' ')
        # count += 1
        # if count == 20:
        #     break
        
        samples = samples.to(device)
        img_h, img_w = samples.tensors.shape[-2:]

        # inference
        outputs = model(samples, test=True, targets=targets, criterion=criterion, eval_s=True)
        outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
        outputs_points = outputs["pred_points"][0]
        outputs_offsets = outputs["pred_offsets"][0]
        outputs_queries = outputs["points_queries"]

        # process predicted points
        predict_cnt = len(outputs_scores)
        gt_cnt = targets[0]["points"].shape[0]

        # compute error
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        abs_ = abs(predict_cnt - gt_cnt) / gt_cnt
        
        if gt_cnt > gt_determined:
            mae_ac = abs(predict_cnt - gt_cnt)
            mse_ac = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
            abs_ac = abs(predict_cnt - gt_cnt) / gt_cnt

        # target boxes
        # print('\n\nargs is \n\n', args.dataset_file, '\n\n')  # dataset_file='RTC'
        if args.dataset_file != "RTC":
            target_set = targets[0]["points"]
            outputs_set = outputs["pred_points"][0]
            outputs_set[:, 0] *= img_h
            outputs_set[:, 1] *= img_w
            target_box = targets[0]["bboxs"]

            ind = outputs["ind"][0]

            if outputs_set.shape[0] != 0:
                tp = cal_distance(target_box, ind, target_set, outputs_set)
                if gt_cnt > gt_determined:
                    tp_ac = tp
            else:
                tp = 0

            # F1 compute
            fn = gt_cnt - tp
            fp = predict_cnt - tp
            Prec = tp / (tp + fp + 1e-8)
            Recall = tp / (tp + fn + 1e-8)
            F1_s = 2 * (Prec * Recall) / (Prec + Recall + 1e-8)
            
            if gt_cnt > gt_determined:
                fn_ac = gt_cnt - tp_ac
                fp_ac = predict_cnt - tp_ac
                pre_ac = tp_ac / (tp_ac + fp_ac + 1e-8)
                rec_ac = tp_ac / (tp_ac + fn_ac + 1e-8)
                f1_ac = 2 * (pre_ac * rec_ac) / (pre_ac + rec_ac + 1e-8)

        else:
            # print('RTC no computing F1')
            Prec, Recall, F1_s, pre_ac, rec_ac, f1_ac = 0, 0, 0, 0, 0, 0

        # record results
        results = {}
        toTensor = lambda x: torch.tensor(x).float().cuda()
        # results['mae'], results['mse'] = toTensor(mae), toTensor(mse)
        (
            results["mae"],
            results["mse"],
            results["Prec"],
            results["Recall"],
            results["F1_s"],
            results["abs"]
        ) = (toTensor(mae), toTensor(mse), Prec, Recall, F1_s, abs_)
        
        if gt_cnt > gt_determined:
            (   
                results["mae_ac"],
                results["mse_ac"],
                results["pre_ac"],
                results["reca_ac"],
                results["f1_ac"],
                results["abs_ac"]
            ) = (toTensor(mae_ac), toTensor(mse_ac), pre_ac, rec_ac, f1_ac, abs_ac)
        # results['gt_cnt'], results['pd_cnt'] = toTensor(gt_cnt), toTensor(predict_cnt)

        if distributed:
            # results = utils.reduce_dict(results)
            gt_cnt_all += [i.cpu().numpy() for i in utils.all_gather(toTensor(gt_cnt))]
            pd_cnt_all += [i.cpu().numpy() for i in utils.all_gather(toTensor(predict_cnt))]
            
            if gt_cnt > gt_determined:
                gt_cnt_all_ac += [i.cpu().numpy() for i in utils.all_gather(toTensor(gt_cnt))]
                pd_cnt_all_ac += [i.cpu().numpy() for i in utils.all_gather(toTensor(predict_cnt))]
            # print('gt_cnt:',gt_cnt, 'gt_cnt_gather:', len(gt_cnt_all))
            metric_logger.update(
                mae=results["mae"],
                mse=results["mse"],
            )
        else:
            if gt_cnt > gt_determined:
                gt_cnt_all_ac.append(gt_cnt)
                pd_cnt_all_ac.append(predict_cnt)
                metric_logger.update(
                    mae=results["mae"],
                    mse=results["mse"],
                    mae_ac=results["mae_ac"],
                    mse_ac=results["mse_ac"],
                )
            else:
                gt_cnt_all.append(gt_cnt)
                pd_cnt_all.append(predict_cnt)
                metric_logger.update(
                    mae=results["mae"],
                    mse=results["mse"],
                )

        # results = utils.reduce_dict(results)
        if gt_cnt > gt_determined:
            metric_logger.update(
                mae=results["mae"],
                mse=results["mse"],
                Prec=results["Prec"],
                Recall=results["Recall"],
                F1_s=results["F1_s"],
                abs=results["abs"],
                
                mae_ac = results["mae_ac"],
                mse_ac=results["mse_ac"],
                pre_ac=results["pre_ac"],
                rec_ac=results["reca_ac"],
                f1_ac=results["f1_ac"],
                abs_ac=results["abs_ac"],
            )
        else:
            metric_logger.update(
                mae=results["mae"],
                mse=results["mse"],
                Prec=results["Prec"],
                Recall=results["Recall"],
                F1_s=results["F1_s"],
                abs=results["abs"],
            )
            

        # visualize predictions
        if vis_dir:
            points = [
                [point[0], point[1]] for point in outputs_points
            ]  # recover to actual points
            split_map = (
                (outputs["split_map_raw"][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
            )
            
            visualization(
                samples, targets, [points], outputs_queries, vis_dir, split_map=split_map, gt_cnt=gt_cnt
            )
            # if gt_cnt > gt_determined:
            #     visualization(
            #         samples, targets, [points], outputs_queries, vis_dir_ac, split_map=split_map,
            #     ) 

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    gt_cnt_array, pd_cnt_array = np.array(gt_cnt_all), np.array(pd_cnt_all)
    gt_cnt_array_ac, pd_cnt_array_ac = np.array(gt_cnt_all_ac), np.array(pd_cnt_all_ac)
    
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results["mse"] = np.sqrt(results["mse"])
    results["mse_ac"] = np.sqrt(results["mse_ac"])
    
    # print(len(gt_cnt_array), len(pd_cnt_array))
    results["r2"] = r2_score(gt_cnt_array, pd_cnt_array)
    results["r2_ac"] = r2_score(gt_cnt_array_ac, pd_cnt_array_ac)
    # results["rmae"], results["rmse"] = metrics.compute_relerr(
    #     gt_cnt_array, pd_cnt_array
    # )
    results["racc"] = metrics.compute_racc(gt_cnt_array, pd_cnt_array)
    results["rac_ac"] = metrics.compute_racc(gt_cnt_array_ac, pd_cnt_array_ac)
    
    order = ["mae", "mae_ac",
             "mse", "mse_ac",
             "Prec", "pre_ac",
             "Recall", "rec_ac",
             "F1_s", "f1_ac",
             "abs", "abs_ac",
             "r2", "r2_ac",
             "racc", "rac_ac"]
    def custom_sort_key(item):
        key, value = item
        index = order.index(key) if key in order else len(order) + 1
        return (index, not key.endswith('_ac'), key)
    # results = dict(sorted(results.items(), key=custom_sort_key))
    
    return dict(sorted(results.items(), key=custom_sort_key))


def cal_distance(sig_s, ind, target_set, outputs_set):
    tp = 0
    for i in range(len(ind[0])):
        
        img_w1 = sig_s[ind[1][i]][0]
        img_h1 = sig_s[ind[1][i]][1]
        img_w2 = sig_s[ind[1][i]][2]
        img_h2 = sig_s[ind[1][i]][3]
        
        img_h = img_h2 - img_h1
        img_w = img_w2 - img_w1
        
        dis = math.sqrt(img_h * img_h + img_w * img_w) / 2
        if (target_set[ind[1][i]][0] - outputs_set[ind[0][i]][0]) * (target_set[ind[1][i]][0] - outputs_set[ind[0][i]][0]) \
            + (target_set[ind[1][i]][1] - outputs_set[ind[0][i]][1]) * (target_set[ind[1][i]][1] - outputs_set[ind[0][i]][1]) \
            <= dis * dis:
            tp += 1
    return tp
