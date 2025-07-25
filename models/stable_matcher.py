import torch
from torch import nn

# --- Gale-Shapley 算法实现 ---
def stable_matching_algorithm(proposer_preferences, reviewer_preferences):
    """
    Gale-Shapley 算法的实现.
    :param proposer_preferences: list[list]，求婚方的偏好列表. proposer_preferences[i] 是求婚者i的偏好排序.
    :param reviewer_preferences: list[list]，被求婚方的偏好列表.
    :return: dict，一个从求婚者到其匹配的对象的字典.
    """
    num_proposers = len(proposer_preferences)
    num_reviewers = len(reviewer_preferences)

    # 将被求婚方的偏好列表转换为排名字典，以便快速查找 O(1)
    # reviewer_rankings[reviewer_id][proposer_id] = rank
    reviewer_rankings = [[0] * num_proposers for _ in range(num_reviewers)]
    for r_id, prefs in enumerate(reviewer_preferences):
        for rank, p_id in enumerate(prefs):
            reviewer_rankings[r_id][p_id] = rank

    # 初始时，所有求婚者都是自由的
    free_proposers = list(range(num_proposers))
    # 记录每个求婚者下一次要求婚的对象索引
    next_proposal_idx = [0] * num_proposers
    # 记录被求婚方当前的匹配对象，-1表示没有
    current_matches = [-1] * num_reviewers

    while free_proposers:
        proposer_id = free_proposers.pop(0)
        
        # 获取该求婚者的下一个偏好对象
        pref_list = proposer_preferences[proposer_id]
        if next_proposal_idx[proposer_id] >= len(pref_list):
            continue # 没有更多可以求婚的对象了
            
        reviewer_id = pref_list[next_proposal_idx[proposer_id]]
        next_proposal_idx[proposer_id] += 1

        current_partner_id = current_matches[reviewer_id]

        if current_partner_id == -1:
            # 如果被求婚方是自由的，则他们暂时匹配
            current_matches[reviewer_id] = proposer_id
        else:
            # 如果被求婚方已经有伴侣，比较新旧求婚者的排名
            rank_of_new_proposer = reviewer_rankings[reviewer_id][proposer_id]
            rank_of_current_partner = reviewer_rankings[reviewer_id][current_partner_id]

            if rank_of_new_proposer < rank_of_current_partner:
                # 新的求婚者排名更高（值更小），接受新的
                current_matches[reviewer_id] = proposer_id
                # 旧的伴侣恢复自由身
                free_proposers.append(current_partner_id)
            else:
                # 新的求婚者排名较低，被拒绝，需要再次求婚
                free_proposers.append(proposer_id)

    matches_dict = {p: r for r, p in enumerate(current_matches) if p != -1}
    return matches_dict

class StableMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, **kwargs):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_points = outputs["pred_points"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_points = torch.cat([v["points"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        
        img_h, img_w = outputs['img_shape']
        out_points_abs = out_points.clone()
        out_points_abs[:,0] *= img_h
        out_points_abs[:,1] *= img_w
        
        cost_point = torch.cdist(out_points_abs, tgt_points.to(out_points_abs.device), p=2)
        
        C = self.cost_point * cost_point + self.cost_class * cost_class
        try:
            C = C.view(bs, num_queries, -1).cpu()
        except:
            bs = len(targets)
            num_queries = out_points_abs.size(0)
            num_targets = tgt_points.size(0)
            C = torch.zeros((bs, num_queries, num_targets), device=out_points_abs.device).cpu()

        sizes = [len(v["points"]) for v in targets]
        
        indices = []

        for i, c in enumerate(C.split(sizes, -1)):
            cost_matrix = c[i]
            num_preds = cost_matrix.shape[0]
            num_targets_in_sample = cost_matrix.shape[1]

            if num_targets_in_sample == 0:
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue

            # 步骤 2: 从代价矩阵生成偏好列表
            # 代价越低，偏好越高
            
            # 预测方的偏好列表（求婚方）
            # 对每一行（每个预测）排序，得到目标的索引
            pred_preferences = torch.argsort(cost_matrix, dim=1).tolist()
            
            # 目标方的偏好列表（被求婚方）
            # 对每一列（每个目标）排序，得到预测的索引
            target_preferences = torch.argsort(cost_matrix, dim=0).T.tolist()

            # 步骤 3: 执行 Gale-Shapley 算法
            # 在此场景中，预测方(queries)数量通常远大于目标方(targets)
            # 让数量较少的一方（targets）作为求婚方可以稍微提高效率，但为保持概念清晰，我们让预测方求婚
            matches_dict = stable_matching_algorithm(pred_preferences, target_preferences)
            
            # 步骤 4: 格式化输出，与原接口保持一致
            if not matches_dict:
                indices.append((torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)))
                continue

            # 将匹配字典转换为 (index_i, index_j) 的格式
            # 注意：字典的 key 是求婚者(pred)，value 是被求婚者(target)
            preds_i = torch.tensor(list(matches_dict.keys()), dtype=torch.int64)
            targets_j = torch.tensor(list(matches_dict.values()), dtype=torch.int64)
            indices.append((preds_i, targets_j))
            
        return indices
    
def build_stable_matcher(args):
    print('\nStable Matcher built!\n')
    return StableMatcher(cost_class=args.set_cost_class, cost_point=args.set_cost_point)