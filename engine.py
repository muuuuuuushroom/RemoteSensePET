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




def visualization(samples, targets, pred, queries, vis_dir, gt_cnt, split_map=None, args=None, outputs=None):
    """                         # pred point & query
    Visualize predictions
    """
    box_flag = True if args.opt_query_decoder else False
    
    if args.dataset_file != 'WuhanMetro':
        vis_dir_d = os.path.join(vis_dir, 'dense')
        vis_dir_nac = os.path.join(vis_dir, 'sparse')
        os.makedirs(vis_dir_d, exist_ok=True)
        os.makedirs(vis_dir_nac, exist_ok=True)
    else:
        vis_dir_d = vis_dir
        vis_dir_nac = vis_dir
        os.makedirs(vis_dir + '/3-1', exist_ok=True)
        os.makedirs(vis_dir + '/3-2', exist_ok=True)
    
    if box_flag:
        vis_dir_box = os.path.join(vis_dir, 'box')
        os.makedirs(vis_dir_box, exist_ok=True)
        # vis_dir_box_nosame = os.path.join(vis_dir, 'box_nosame')
        # os.makedirs(vis_dir_box_nosame, exist_ok=True)
    
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
        or_sample = sample_vis
        h, w = sample_vis.shape[:2]
        # sample_vis = cv2.resize(sample_vis, (512, 512))

        if box_flag:
            pq_ex = outputs['pq_ex']
            sample_vis_box = or_sample
            batch_size, num_points, _ = pq_ex.shape
            
            fixed_colors = [
                # (0, 0, 255),      # 红
                # (0, 128, 255),    # 橙
                # (0, 255, 255),    # 黄
                (0, 255, 0),      # 绿
                # (255, 255, 0),    # 青
                (255, 0, 0),      # 蓝
                (255, 0, 255),    # 紫
            ]
            
            # for i, p in enumerate(pred[idx]):
            #     sample_vis_box = cv2.circle(
            #         sample_vis_box, (int(p[1]), int(p[0])), 4, (255, 0, 255), -1
            #         )
            for i, t in enumerate(gts[idx]):
                sample_vis_box = cv2.circle(
                    sample_vis_box, (int(t[1]), int(t[0])), 3, (0, 0, 255), -1
                )
            
            for i in range(num_points):  # 遍历每个点序号
                for b in range(batch_size):  # 每个 batch 的该点
                    q = pq_ex[b][i].clone()
                    q[1] *= w
                    q[0] *= h
                    q_x, q_y = int(q[1]), int(q[0])
                    color = fixed_colors[b % len(fixed_colors)]
                    sample_vis_box = cv2.circle(sample_vis_box, (q_x, q_y), 2, color, -1)
                    
            for i, p in enumerate(pred[idx]):
                sample_vis_box = cv2.circle(
                    sample_vis_box, (int(p[1]), int(p[0])), 2, (255, 0, 255), -1
                    )

        
            name = targets[idx]["image_path"].split("/")[-1].split(".")[0]
            # start_points = pq_ex[0]      # shape: [num_points, 2]
            # end_points   = pq_ex[-1]
            # offsets = end_points - start_points
            # if offsets.shape[0] != 0:
            # ref_offset = offsets[0]
            # is_equal = torch.all(offsets == ref_offset, dim=1) 
            # same_ratio = is_equal.float().mean().item()
            save_path = os.path.join(
                    vis_dir_box,
                    "{}_gt{}_pred{}.jpg".format(name, len(gts[idx]), len(pred[idx])),
                ) 
            cv2.imwrite(
                save_path,
                sample_vis_box,
            )
            
        # if not box_flag:
        # draw ground-truth points (red)
        # if '1208' in targets[idx]["image_path"].split("/")[-1].split(".")[0]:
        #     mul = 2
        # else:
        #     mul = 1
        mul = 1
        if '970' in targets[idx]["image_path"].split("/")[-1].split(".")[0]:
            continue
            
        size = 2
        for i, t in enumerate(gts[idx]):
            sample_vis = cv2.circle(
                sample_vis, (int(t[1]*mul), int(t[0])*mul), size+2, (0, 0, 255), -1
            )
                    # cv2.putText(sample_vis, str(i+1), (int(t[1]+3), int(t[0])+3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # draw predictions (green)
        for i, p in enumerate(pred[idx]):
            sample_vis = cv2.circle(
                sample_vis, (int(p[1]), int(p[0])), size+1, (0, 255, 0), -1
                )
                    # cv2.putText(sample_vis, str(i+1), (int(p[1]+3), int(p[0])+3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # draw point-query
        # for i, q in enumerate(queries[idx]):
            
        #     if args.predict == 'origin':
        #         q[1] *= w
        #         q[0] *= h
        #     sample_vis = cv2.circle(
        #         sample_vis, (int(q[1]), int(q[0])), size, (0, 255, 255), -1
        #         )
        #     # draw line between query and pred
        #     q_x, q_y = int(q[1]), int(q[0])
        #     p_x, p_y = int(pred[idx][i][1]), int(pred[idx][i][0])
        #     overlay = sample_vis.copy()
        #     cv2.line(overlay, (p_x, p_y), (q_x, q_y), (0, 255, 0), 2) 
        #     alpha = 0.5 
        #     sample_vis = cv2.addWeighted(overlay, alpha, sample_vis, 1 - alpha, 0)
            # cv2.putText(sample_vis, str(i+1), (int(q[1]*w-3), int(q[0]*h)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # draw split map
        # if split_map is not None:
        #     imgH, imgW = sample_vis.shape[:2]
        #     split_map = (split_map * 255).astype(np.uint8)
        #     split_map = 255 - split_map
        #     split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
        #     split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
        #     sample_vis = split_map * 0.7 + sample_vis

        # save image
        if vis_dir is not None:
            # eliminate invalid area
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[: valid_h + 1, : valid_w + 1]
            

        # --- V V V --- 帮助函数：添加带透明背景的文本 (修改为左上角) --- V V V ---
            def add_text_with_transparent_bg(image, w_img, h_img):
                # 文本内容
                gt_text = f"GT: {len(gts[idx])}"
                pred_text = f"Pred: {len(pred[idx])}"
                
                # 字体和颜色设置
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                text_color = (0, 0, 0)      # 黑色字体
                bg_color = (255, 255, 255) # 白色背景
                
                # 透明度: 70% 透明 -> 30% 不透明
                alpha = 0.3 
                beta = 1.0 - alpha
                
                # 边距和行距
                padding = 5  # 文本和矩形框边缘的内部边距
                line_spacing = 5 # 两行文本之间的额外间距
                
                # 1. 计算文本大小和矩形框位置
                (gt_text_w, gt_text_h), gt_baseline = cv2.getTextSize(gt_text, font, font_scale, thickness)
                (pred_text_w, pred_text_h), pred_baseline = cv2.getTextSize(pred_text, font, font_scale, thickness)
                
                max_text_w = max(gt_text_w, pred_text_w)
                # 注意: gt_text_h 是从基线到文本顶部的距离
                line_height = gt_text_h # 我们使用这个作为基线到基线的距离
                
                # 矩形框坐标 (紧贴左上角 0,0)
                rect_x1 = 0
                rect_y1 = 0
                rect_x2 = padding + max_text_w + padding
                # 总高度 = 顶部padding + 第一行高度 + 行距 + 第二行高度 + 底部padding
                # (我们用 gt_text_h 近似两行的高度)
                rect_y2 = padding + gt_text_h + line_spacing + pred_text_h + padding + pred_baseline
                
                # 确保坐标不会超出图像边界
                rect_x1 = max(0, rect_x1)
                rect_y1 = max(0, rect_y1)
                rect_x2 = min(w_img, rect_x2)
                rect_y2 = min(h_img, rect_y2)

                # 2. 创建一个副本用于混合
                overlay = image.copy()
                # 在副本上绘制实心白色矩形
                cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
                
                # 3. 将副本 (带矩形) 与原始图像混合
                blended_image = cv2.addWeighted(overlay, alpha, image, beta, 0.0)
                
                # 4. 在混合后的图像上绘制黑色文本 (实现左对齐)
                # GT 文本
                gt_text_x = padding
                gt_text_y = padding + gt_text_h # (x, y) 是基线的左下角
                cv2.putText(blended_image, gt_text, (gt_text_x, gt_text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
                
                # Pred 文本
                pred_text_x = padding
                # 第二行的基线 = 第一行的基线 + 行高 + 行距
                pred_text_y = gt_text_y + gt_text_h + line_spacing 
                cv2.putText(blended_image, pred_text, (pred_text_x, pred_text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
                
                return blended_image
            # --- ^ ^ ^ --- 帮助函数结束 --- ^ ^ ^ ---

            # --- V V V --- 在这里添加文本 (用于 sample_vis_box) --- V V V ---
            # 使用原始图像的 w, h
            sample_vis = add_text_with_transparent_bg(sample_vis, w, h)
            # --- ^ ^ ^ --- 添加文本结束 --- ^ ^ ^ ---
            
            if args.dataset_file == 'WuhanMetro':
                pre_name_folder = targets[idx]["image_path"].split("/")[-2]
            name = targets[idx]["image_path"].split("/")[-1].split(".")[0]
            
            if gt_cnt > 100:
                cv2.imwrite(
                    os.path.join(
                        vis_dir_d,
                        "{}_gt{}_pred{}.jpg".format(name, len(gts[idx]), len(pred[idx])),
                    ),
                    sample_vis,
                )
            else:
                if args.dataset_file == 'WuhanMetro':
                    vis_dir_nac = os.path.join(vis_dir, pre_name_folder)
                cv2.imwrite(
                    os.path.join(
                        vis_dir_nac,
                        "{}_gt{}_pred{}.jpg".format(name, len(gts[idx]), len(pred[idx])),
                    ),
                    sample_vis,
                )


def _to_device(v, device):
    if torch.is_tensor(v):
        return v.to(device, non_blocking=True)
    # 兼容 list/tuple 里嵌 tensor 的情况（可选）
    if isinstance(v, (list, tuple)):
        return type(v)(_to_device(x, device) for x in v)
    return v

# training
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets, probability in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        # probability adapatation
        if probability[0] is not None:
            probability = probability.to(device)
        
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: _to_device(v, device) for k, v in t.items()} for t in targets]
        gt_points = [target['points'] for target in targets]
        
        # start training transformer
        outputs = model(samples, epoch=epoch, train=True, 
                        criterion=criterion, targets=targets,
                        probability=probability)
        loss_dict, weight_dict, losses = outputs['loss_dict'], outputs['weight_dict'], outputs['losses']

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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


# evaluation
@torch.no_grad()
def evaluate(model, data_loader, device, epoch=0, vis_dir=None, distributed=False, args=None, criterion=None): 
    # for more VRAM, use: 
    torch.cuda.empty_cache()
    
    model.eval()
    gt_determined = 1 if args.dataset_file == 'WuhanMetro' else 1000000
    # gt_determined = 100
    metric_logger = utils.MetricLogger(delimiter="  ", win_size=len(data_loader))
    header = 'Test:'

    # if vis_dir is not None:
        # os.makedirs(vis_dir, exist_ok=True)
        # vis_dir_d = os.path.join(vis_dir, 'gt_d')
        # os.makedirs(vis_dir_d, exist_ok=True)
        
    gt_cnt_all, pd_cnt_all = [], []
    gt_cnt_all_d, pd_cnt_all_d = [], []
    print_freq = 10; count = 0

    for samples, targets, prob in metric_logger.log_every(data_loader, print_freq, header):
        
        samples = samples.to(device)    # tensors [torch.Size([1, 3, 512, 1024])]
        img_h, img_w = samples.tensors.shape[-2:]     
        outputs = model(samples, test=True, targets=targets, criterion=criterion, eval_s=True)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        outputs_offsets = outputs['pred_offsets'][0]
        outputs_queries = outputs['points_queries']

        # process predicted points
        predict_cnt = len(outputs_scores)
        gt_cnt = targets[0]['points'].shape[0]

        # compute error
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        abs_ = abs(predict_cnt - gt_cnt) / gt_cnt
        
        if gt_cnt > gt_determined:
            mae_d = abs(predict_cnt - gt_cnt)
            mse_d = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
            abs_d = abs(predict_cnt - gt_cnt) / gt_cnt

        # target boxes
        # print('\n\nargs is \n\n', args.dataset_file, '\n\n')  # dataset_file='RTC'
        if args.dataset_file in ['Ship', 'People', 'Car']:
            target_set = targets[0]["points"]
            outputs_set = outputs["pred_points"][0]
            outputs_set[:, 0] *= img_h
            outputs_set[:, 1] *= img_w
            target_box = targets[0]["bboxs"]

            ind = outputs["ind"][0]

            if outputs_set.shape[0] != 0:
                tp = cal_distance(target_box, ind, target_set, outputs_set)
                tp_d = tp
            else:
                tp_d = 0
                tp = 0
                if gt_cnt > gt_determined:
                    tp_d = tp

            # F1 compute
            fn = gt_cnt - tp
            fp = predict_cnt - tp
            Prec = tp / (tp + fp + 1e-8)
            Recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (Prec * Recall) / (Prec + Recall + 1e-8)
            
            if gt_cnt > gt_determined:
                fn_d = gt_cnt - tp_d
                fp_d = predict_cnt - tp_d
                pre_d = tp_d / (tp_d + fp_d + 1e-8)
                rec_d = tp_d / (tp_d + fn_d + 1e-8)
                f1_d = 2 * (pre_d * rec_d) / (pre_d + rec_d + 1e-8)


        # record results
        results = {}
        # toTensor = lambda x: torch.tensor(x).float().cuda()
        toTensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)

        results['mae'], results['mse'] = toTensor(mae), toTensor(mse)
        
        if args.dataset_file in ['Ship', 'People', 'Car']:
            results["Prec"], results["Recall"], results["f1"], results["abs"] = Prec, Recall, f1, abs_
            if gt_cnt > gt_determined:
                (   
                    results["mae_d"],
                    results["mse_d"],
                    results["pre_d"],
                    results["reca_d"],
                    results["f1_d"],
                    results["abs_d"]
                ) = (toTensor(mae_d), toTensor(mse_d), pre_d, rec_d, f1_d, abs_d)
        else:
            results["Prec"], results["Recall"], results["f1"], results["abs"]= 0, 0, 0, 0
            results["mae_d"], results["mse_d"], results["pre_d"], results["reca_d"], results["f1_d"], results["abs_d"] = 0, 0, 0, 0, 0, 0
        
        if distributed:
            # 关键：不要把标量转 CUDA tensor 再 all_gather
            gathered_gt = utils.all_gather(int(gt_cnt))
            gathered_pd = utils.all_gather(int(predict_cnt))

            if utils.is_main_process():
                gt_cnt_all += gathered_gt
                pd_cnt_all += gathered_pd

            if gt_cnt > gt_determined:
                gathered_gt_d = utils.all_gather(int(gt_cnt))
                gathered_pd_d = utils.all_gather(int(predict_cnt))
                if utils.is_main_process():
                    gt_cnt_all_d += gathered_gt_d
                    pd_cnt_all_d += gathered_pd_d

            metric_logger.update(mae=results["mae"], mse=results["mse"])
        else:
            if gt_cnt > gt_determined:
                gt_cnt_all_d.append(gt_cnt)
                pd_cnt_all_d.append(predict_cnt)
                metric_logger.update(
                    mae=results["mae"],
                    mse=results["mse"],
                    mae_d=results["mae_d"],
                    mse_d=results["mse_d"],
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
                f1=results["f1"],
                abs=results["abs"],
                
                mae_d = results["mae_d"],
                mse_d=results["mse_d"],
                pre_d=results["pre_d"],
                rec_d=results["reca_d"],
                f1_d=results["f1_d"],
                abs_d=results["abs_d"],
            )
        else:
            metric_logger.update(
                mae=results["mae"],
                mse=results["mse"],
                Prec=results["Prec"],
                Recall=results["Recall"],
                f1=results["f1"],
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
                samples, targets, [points], outputs_queries, vis_dir, 
                split_map=split_map, gt_cnt=gt_cnt, args=args, outputs=outputs
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if distributed and (not utils.is_main_process()):
        results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        results["mse"] = math.sqrt(results["mse"])
        # r2/rmae/rmse/racc 在非主进程不计算，避免依赖空数组
        results["r2"] = float("nan")
        results["rmae"] = float("nan")
        results["rmse"] = float("nan")
        results["racc"] = float("nan")
        return results
    gt_cnt_array, pd_cnt_array = np.array(gt_cnt_all), np.array(pd_cnt_all)
    gt_cnt_array_d, pd_cnt_array_d = np.array(gt_cnt_all_d), np.array(pd_cnt_all_d)
    
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results["mse"] = np.sqrt(results["mse"])
    # results["mse_d"] = np.sqrt(results["mse_d"])
    
    # print(len(gt_cnt_array), len(pd_cnt_array))
    results["r2"] = r2_score(gt_cnt_array, pd_cnt_array) if args.dataset_file != 'WuhanMetro' else r2_score(gt_cnt_array_d, pd_cnt_array_d)
    # results["r2_d"] = r2_score(gt_cnt_array_d, pd_cnt_array_d)
    # results["rmae"], results["rmse"] = metrics.compute_relerr(
    #     gt_cnt_array, pd_cnt_array
    # )
    results['rmae'], results['rmse'] = metrics.compute_relerr(gt_cnt_array, pd_cnt_array)
    results["racc"] = metrics.compute_racc(gt_cnt_array, pd_cnt_array)
    # results["rac_d"] = metrics.compute_racc(gt_cnt_array_d, pd_cnt_array_d)
    
    if args.dataset_file in ['Ship', 'People', 'Car']:
        results["mse_d"] = np.sqrt(results["mse_d"])
        results["r2_d"] = r2_score(gt_cnt_array_d, pd_cnt_array_d)
        results["rac_d"] = metrics.compute_racc(gt_cnt_array_d, pd_cnt_array_d)
        order = ["mae", "mae_d",
                "mse", "mse_d",
                "Prec", "pre_d",
                "Recall", "rec_d",
                "f1", "f1_d",
                "abs", "abs_d",
                "r2", "r2_d",
                "racc", "rac_d",
                "rmae", "rmse"]
        
        def custom_sort_key(item):
            key, value = item
            index = order.index(key) if key in order else len(order) + 1
            return (index, not key.endswith('_d'), key)
        results = dict(sorted(results.items(), key=custom_sort_key))

    return results
