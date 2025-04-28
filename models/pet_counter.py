"""
from: hxn
"""


import torch
import torch.nn.functional as F
from torch import nn
from models.pacpet.Block.Blocks_etop import Cross_Block
from einops import rearrange,repeat
from models.pacpet.matcher import HungarianMatcher
class MLP(nn.Module):
    """
    Multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, is_reduce=False, use_relu=True):
        super().__init__()
        self.num_layers = num_layers
        if is_reduce:
            h = [hidden_dim//2**i for i in range(num_layers - 1)]
        else:
            h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.use_relu = use_relu

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.use_relu:
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            else:
                x = layer(x)
        return x


class BasePETCount(nn.Module):
    """ 
    Base PET model
    """
    def __init__(self, num_classes, quadtree_layer='sparse', level='8x', stride=8, enhance=False, **kwargs):
        super().__init__()
        hidden_dim = 512

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        if enhance:
            self.transformer = nn.ModuleList([Cross_Block()
                                            for i in range(2)])
        else:
            self.transformer = None
        
        # self.level = '4x' if quadtree_layer == 'dense' else '8x'
        # self.pq_stride =  4 if quadtree_layer == 'dense' else 8
        self.level = level
        self.pq_stride =  stride
    
    def points_queris_embed(self, images, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during training
        """
        # dense position encoding at every pixel location
        # dense_input_embed = kwargs['dense_input_embed']
        # bs, c = dense_input_embed.shape[:2]


        # generate point queries
        H, W = images.shape[-2:]
        shift_x = (torch.arange(0,H,stride) + stride//2).long()# TODO check
        shift_y = (torch.arange(0,W,stride) + stride//2).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)

        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2,  for sparse:(384/8)^2 x 2, for dense:(384/4)^2 x 2 
        h, w = shift_x.shape # h = w = 384//stride

        # # get point queries embedding
        # query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        # bs, c = query_embed.shape[:2]
        # query_embed = query_embed.view(bs, c, h, w) # query_embed:8,256,32,3,   is position(image[x,y]) where p(x,y) is the query point

        # get point queries features, equivalent to nearest interpolation
        bs,c,h,w = src.shape
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        points_queries = repeat(points_queries,'n d->b n d',b=bs)

        query_feats = src[:, :, shift_y_down,shift_x_down]
        # query_feats = query_feats.view(bs, c, h, w)

        return  points_queries, query_feats  # query point position embed_vector, query point position in [x,y], featuer vector at query position

    def points_queris_embed_inference(self, images, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during inferencing:according to split map
        """
        # generate point queries
        H, W = images.shape[-2:]

        # NOTE: pin bs = 1 while doing inferece
        bs,c,h,w = src.shape
        split_map = kwargs['div']
        for mask in split_map:
            valid_x, valid_y = torch.where(mask>0.5)
        query_feats = src[:, :, valid_x,valid_y]

        valid_x = self.pq_stride * valid_x + self.pq_stride // 2
        valid_y = self.pq_stride * valid_y + self.pq_stride // 2
        
        points_queries = torch.vstack([valid_x.flatten(), valid_y.flatten()]).permute(1,0) 
        points_queries = repeat(points_queries,'n d->b n d',b=bs)

        return  points_queries, query_feats, (valid_x,valid_y)  # query point position embed_vector, query point position in [x,y], featuer vector at query position
    
    def get_point_query(self, images, context_info:dict, **kwargs):
        """
        Generate point query
        """
        src = context_info[self.level]    # for dense:b 512 48 48

        # generate points queries and position embedding
        if 'train' in kwargs:
            points_queries, query_feats = self.points_queris_embed(images, self.pq_stride, src, **kwargs) # query_embed is actually query_pos_embed
            v_idx = None
        else:
            points_queries, query_feats, v_idx = self.points_queris_embed_inference(images, self.pq_stride, src, **kwargs)

        out = (points_queries, query_feats, v_idx)
        return out
    
    def predict(self, samples, points_queries, hs, **kwargs):
        """
        Crowd prediction
        """
        outputs_class = self.class_embed(hs)
        # normalize to -1~1
        outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0 

        # normalize point-query coordinates
        img_shape = samples.shape[-2:]
        img_h, img_w = img_shape
        points_queries = points_queries.float().to(outputs_class.device)
        points_queries[:, :, 0] /= (self.pq_stride // 2)
        points_queries[:, :, 1] /= (self.pq_stride // 2)

        outputs_points = outputs_offsets + points_queries
        # normalize coords: 0~1
        outputs_points[:,:,0] *= ((self.pq_stride // 2) / img_h) 
        outputs_points[:,:,1] *= ((self.pq_stride // 2) / img_w)

        points_queries[:, :, 0] *= ((self.pq_stride // 2) / img_h) 
        points_queries[:, :, 1] *= ((self.pq_stride // 2) / img_w)

        out = {'pred_logits': outputs_class, 'pred_points': outputs_points, 'img_shape': img_shape, 'pred_offsets': outputs_offsets}
    
        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

    def forward(self,image,feature_map_img,feature_map_exa,**kwargs):  # sample:iamge 256x256, feature:cnn fmap 8x:32x32 4x:64x64, context_info: transformer feature 32x32

        # get points queries for context info
        pqs = self.get_point_query(image, feature_map_img, **kwargs)
        
        # point querying
        kwargs['pq_stride'] = self.pq_stride

        # make cross atten here
        query = pqs[1]
        # query = rearrange(pqs[1], 'b d h w->b (h w) d') pqs[1]:b,d,n
        # examplar = rearrange(feature_map_exa[self.level], 'b d h w->b (h w) d')

        if self.transformer != None:
            for block in self.transformer:
                query, examplar = block(query, examplar)
        else:
            query = rearrange(query,'b d n->b n d')
        # hs = self.transformer(feature_map_img, feature_map_exa, pqs) # hs:bs, 576, 512

        # prediction
        points_queries = pqs[0]
        outputs = self.predict(image, points_queries, query, **kwargs)
        return outputs
    

class SetCriterion(nn.Module):
    """ Compute the loss for PET:
        1) compute hungarian assignment between ground truth points and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction and split map
    """
    def __init__(self, num_classes, matcher:HungarianMatcher, weight_dict, eos_coef, losses):
        """
        Parameters:
            num_classes: one-class in crowd counting
            matcher: module able to compute a matching between targets and point queries
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef    # coefficient for non-object background points
        self.register_buffer('empty_weight', empty_weight)
        # self.div_thrs_dict = {8: 0.0, 4:0.5}
        self.div_thrs_dict = {16: 0.0, 8:0.5}
    
    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):
        """
        Classification loss:
            - targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # compute classification loss
        if 'div' in kwargs:
            # get sparse / dense image index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            ds_idx = den_sort[:len(den_sort)//2]
            sp_idx = den_sort[len(den_sort)//2:]
            eps = 1e-5

            # raw cross-entropy loss
            weights = target_classes.clone().float()
            weights[weights==0] = self.empty_weight[0]
            weights[weights==1] = self.empty_weight[1]
            raw_ce_loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, ignore_index=-1, reduction='none')

            # binarize split map
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs

            # dual supervision for sparse/dense images
            loss_ce_sp = (raw_ce_loss * weights * div_mask)[sp_idx].sum() / ((weights * div_mask)[sp_idx].sum() + eps)
            loss_ce_ds = (raw_ce_loss * weights * div_mask)[ds_idx].sum() / ((weights * div_mask)[ds_idx].sum() + eps)
            loss_ce = loss_ce_sp + loss_ce_ds

            # loss on non-div regions
            non_div_mask = split_map <= div_thrs
            loss_ce_nondiv = (raw_ce_loss * weights * non_div_mask).sum() / ((weights * non_div_mask).sum() + eps)
            loss_ce = loss_ce + loss_ce_nondiv
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=-1)

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, outputs, targets, indices, num_points, **kwargs):
        """
        SmoothL1 regression loss:
           - targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 2]
        """
        assert 'pred_points' in outputs
        # get indices
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # compute regression loss
        losses = {}
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        target_points[:, 0] /= img_h
        target_points[:, 1] /= img_w
        loss_points_raw = F.smooth_l1_loss(src_points, target_points, reduction='none')

        if 'div' in kwargs:
            # get sparse / dense index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            img_ds_idx = den_sort[:len(den_sort)//2]
            img_sp_idx = den_sort[len(den_sort)//2:]
            pt_ds_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_ds_idx])
            pt_sp_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_sp_idx])

            # dual supervision for sparse/dense images
            eps = 1e-5
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs
            loss_points_div = loss_points_raw * div_mask[idx].unsqueeze(-1)
            loss_points_div_sp = loss_points_div[pt_sp_idx].sum() / (len(pt_sp_idx) + eps)
            loss_points_div_ds = loss_points_div[pt_ds_idx].sum() / (len(pt_ds_idx) + eps)

            # loss on non-div regions
            non_div_mask = split_map <= div_thrs
            loss_points_nondiv = (loss_points_raw * non_div_mask[idx].unsqueeze(-1)).sum() / (non_div_mask[idx].sum() + eps)   

            # final point loss
            losses['loss_points'] = loss_points_div_sp + loss_points_div_ds + loss_points_nondiv
        else:
            losses['loss_points'] = loss_points_raw.sum() / num_points
        
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'{loss} loss is not defined'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ Loss computation
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_points)
        # num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))
        return losses

class pet_counter(nn.Module):
    def __init__(self,pet_decoder):
        super().__init__()

        hidden_dim = 769

        self.split = pet_decoder['spliter']
        if pet_decoder['spliter'] == 'latent':
            self.splitter = nn.Sequential(
                nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
                nn.Conv2d(hidden_dim, 256, 3,1,1),
                nn.Conv2d(256, 256, 3,1,1),
                nn.Conv2d(256, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            
        elif pet_decoder['spliter'] == 'attn':
            self.splitter = nn.Sequential(
                nn.Sigmoid()
            )
        self.attn_threshold = 0. # TODO:modify this
        self.quadtree_sparse = BasePETCount(2, quadtree_layer=pet_decoder['quadtree_layer'][0],
                                            level=pet_decoder['level'][0],
                                            stride=pet_decoder['stride'][0])
        self.quadtree_dense = BasePETCount(2, quadtree_layer=pet_decoder['quadtree_layer'][1],
                                            level=pet_decoder['level'][1],
                                            stride=pet_decoder['stride'][1])
        # self.level = pet_decoder['level']

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        weight_dict = criterion.weight_dict
        warmup_ep = 5

        # compute loss
        if epoch >= warmup_ep:
            loss_dict_sparse = criterion(output_sparse, targets, div=outputs['split_map_sparse'])
            loss_dict_dense = criterion(output_dense, targets, div=outputs['split_map_dense'])
        else:
            loss_dict_sparse = criterion(output_sparse, targets)
            loss_dict_dense = criterion(output_dense, targets)

        # sparse point queries loss
        loss_dict_sparse = {k+'_sp':v for k, v in loss_dict_sparse.items()}
        weight_dict_sparse = {k+'_sp':v for k,v in weight_dict.items()}
        loss_pq_sparse = sum(loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse.keys() if k in weight_dict_sparse)

        # dense point queries loss
        loss_dict_dense = {k+'_ds':v for k, v in loss_dict_dense.items()}
        weight_dict_dense = {k+'_ds':v for k,v in weight_dict.items()}
        loss_pq_dense = sum(loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense.keys() if k in weight_dict_dense)
    
        # point queries loss
        losses = loss_pq_sparse + loss_pq_dense 

        # update loss dict and weight dict
        loss_dict = dict()
        loss_dict.update(loss_dict_sparse)
        loss_dict.update(loss_dict_dense)

        weight_dict = dict()
        weight_dict.update(weight_dict_sparse)
        weight_dict.update(weight_dict_dense)

        # quadtree splitter loss
        den = torch.tensor([target['density'] for target in targets])   # crowd density
        bs = len(den)
        ds_idx = den < 2 * self.quadtree_sparse.pq_stride   # dense regions index
        ds_div = outputs['split_map_raw'][ds_idx]
        sp_div = 1 - outputs['split_map_raw']

        # constrain sparse regions
        loss_split_sp = 1 - sp_div.view(bs, -1).max(dim=1)[0].mean()

        # constrain dense regions
        if sum(ds_idx) > 0:
            ds_num = ds_div.shape[0]
            loss_split_ds = 1 - ds_div.view(ds_num, -1).max(dim=1)[0].mean()
        else:
            loss_split_ds = outputs['split_map_raw'].sum() * 0.0

        # update quadtree splitter loss            
        loss_split = loss_split_sp + loss_split_ds
        weight_split = 0.1 if epoch >= warmup_ep else 0.0
        loss_dict['loss_split'] = loss_split
        weight_dict['loss_split'] = weight_split

        # final loss
        if self.split == 'latent':
            losses += loss_split * weight_split
        else:
            loss_dict['loss_split'] = 0.0
            weight_dict['loss_split'] = 0.0
        return {'loss_dict':loss_dict, 'weight_dict':weight_dict, 'losses':losses}
    
    def forward(self, image, feature_map_img:dict, feature_map_exa:dict,  **kwargs):
        # context_infor:
        # 16x:512,24,24 + 512,4,12
        # 8x: 512,48,48 + 512,8,24
        # 4x: 512,96,96 + 512,16,48
        if 'train' in kwargs:
            out = self.train_forward(image, feature_map_img, feature_map_exa, **kwargs)
        else:
            out = self.test_forward(image, feature_map_img, feature_map_exa, **kwargs)   

            
        return out

    
    def train_forward(self,image,feature_map_img,feature_map_exa,**kwargs):

        outputs = self.pet_forward(image,feature_map_img,feature_map_exa,**kwargs)
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        losses = self.compute_loss(outputs, criterion, targets, epoch, image)

        return losses

    def test_forward(self,image,feature_map_img,feature_map_exa,**kwargs):
        outputs = self.pet_forward(image,feature_map_img,feature_map_exa,**kwargs)
        out_dense, out_sparse = outputs['dense'], outputs['sparse']
        thrs = 0.5  # inference threshold        
        
        # process sparse point queries
        if outputs['sparse'] is not None:
            out_sparse_scores = torch.nn.functional.softmax(out_sparse['pred_logits'], -1)[..., 1]
            valid_sparse = out_sparse_scores > thrs
            index_sparse = valid_sparse.cpu()
        else:
            index_sparse = None

        # process dense point queries
        if outputs['dense'] is not None:
            out_dense_scores = torch.nn.functional.softmax(out_dense['pred_logits'], -1)[..., 1]
            valid_dense = out_dense_scores > thrs
            index_dense = valid_dense.cpu()
        else:
            index_dense = None

        # format output
        div_out = dict()
        output_names = out_sparse.keys() if out_sparse is not None else out_dense.keys()
        for name in list(output_names):
            if 'pred' in name:
                if index_dense is None:
                    div_out[name] = out_sparse[name][index_sparse].unsqueeze(0)
                elif index_sparse is None:
                    div_out[name] = out_dense[name][index_dense].unsqueeze(0)
                else:
                    div_out[name] = torch.cat([out_sparse[name][index_sparse].unsqueeze(0), out_dense[name][index_dense].unsqueeze(0)], dim=1)
            else:
                div_out[name] = out_sparse[name] if out_sparse is not None else out_dense[name]
        div_out['split_map_raw'] = outputs['split_map_raw']
        return div_out  

    def pet_forward(self,image,feature_map_img,feature_map_exa,**kwargs):
        try:
            b, c, h, w = image.shape
        except:
            c, h, w = image.shape
        h_8x, w_8x = int(h // 8), int(w // 8)
        h_4x, w_4x = int(h // 4), int(w // 4)
        # h_8x, w_8x = int(h // 16), int(w // 16)
        # h_4x, w_4x = int(h // 8), int(w // 8)
        split_map = self.splitter(feature_map_img[self.split])
        split_map_dense = F.interpolate(split_map, (h_4x, w_4x)).reshape(b, -1)
        split_map_sparse = 1 - F.interpolate(split_map,(h_8x, w_8x)).reshape(b, -1)


        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:
            kwargs['div'] = split_map_sparse.reshape(b, h_8x, w_8x)
            outputs_sparse = self.quadtree_sparse(image,feature_map_img,feature_map_exa, **kwargs)
        else:
            outputs_sparse = None
        
        # quadtree layer1 forward (dense)
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            kwargs['div'] = split_map_dense.reshape(b, h_4x, w_4x)
            outputs_dense = self.quadtree_dense(image,feature_map_img,feature_map_exa, **kwargs)
        else:
            outputs_dense = None
        # output_sparse = self.quadtree_sparse(image,feature_map_img,feature_map_exa,split_map_sparse) # output_sparse: 48x48,512->48x48,3
        # output_dense = self.quadtree_dense(image,feature_map_img,feature_map_exa,split_map_dense)    # output_dense:  96x96,512->96x96,3

        outputs = dict()
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        return outputs
        

        


def build_counter(counter_config):
    return pet_counter(counter_config)