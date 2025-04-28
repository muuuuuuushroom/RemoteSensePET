"""
Modules to compute bipartite matching
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def generate_probability_map(points, H, W, sigma):
    """
    probability map generation
    """
    device = points.device
    prob_map = torch.zeros((H, W), device=device)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid_x = grid_x.float()
    grid_y = grid_y.float()

    for point in points:
        mu_x, mu_y = point[0], point[1]
        gaussian = torch.exp(-((grid_x - mu_x) ** 2 + (grid_y - mu_y) ** 2) / (2 * sigma ** 2))
        prob_map = torch.maximum(prob_map, gaussian)
        
    return prob_map

class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network
    """
    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_point: This is the relative weight of the L2 error of the point coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, **kwargs):
        """ 
        Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, 2] with the classification logits
                 "pred_points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_points] (where num_target_points is the number of ground-truth
                           objects in the target) containing the class labels
                 "points": Tensor of dim [num_target_points, 2] containing the target point coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, 2]
        out_points = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # concat target labels and points
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_points = torch.cat([v["points"] for v in targets])

        # compute the classification cost, i.e., - prob[target class]
        cost_class = -out_prob[:, tgt_ids]

        # compute the L2 cost between points
        img_h, img_w = outputs['img_shape']
        out_points_abs = out_points.clone()
        device = out_points_abs.device
        
        out_points_abs[:,0] *= img_h
        out_points_abs[:,1] *= img_w
        
        # d2cnet: probablity map
        # it is not implemented like this
        # opt_loss = 'normal'
        # # opt_loss = 'normal'
        
        # if opt_loss == 'probablity':
        #     sigma = 3
        #     pred_prob_map = generate_probability_map(out_points_abs, img_h, img_w, sigma)
        #     gt_prob_map = generate_probability_map(tgt_points.to(device), img_h, img_w, sigma)
        #     diff = pred_prob_map - gt_prob_map
        #     cost_point = (diff ** 2).mean()
        # else:
        #     cost_point = torch.cdist(out_points_abs, tgt_points.to(out_points_abs.device), p=2)
        
        cost_point = torch.cdist(out_points_abs, tgt_points.to(out_points_abs.device), p=2)
        
        # final cost matrix
        C = self.cost_point * cost_point + self.cost_class * cost_class
        # C = C.view(bs, num_queries, -1).cpu()
   
        # check C shape and numel    
        try:
            C = C.view(bs, num_queries, -1).cpu()
        except:
            bs = len(targets)
            num_queries = out_points_abs.size(0)
            num_targets = tgt_points.size(0)
            C = torch.zeros((bs, num_queries, num_targets), device=out_points_abs.device).cpu()

        sizes = [len(v["points"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_point=args.set_cost_point)


if __name__ == "__main__":
    macher = HungarianMatcher()

    outputs = {
        'pred_points': torch.rand((8, 4096, 2)),
        'pred_logits': torch.rand(8, 4096, 2),
        'img_shape': (512, 512)
    }

    targets = [{'points':torch.rand(36, 2), 'labels':torch.ones(36, dtype=torch.int64)} for i in range(8)]
    import time
    st = time.time()
    indice = macher(outputs, targets)
    print(f'use time: {time.time() - st}')
    # print(indice)
