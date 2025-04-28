"""
Transformer window-rize tools
"""

def enc_win_partition(src, pos_embed, mask, enc_win_h, enc_win_w):  # 4,8
    """
    window-rize input for encoder
    """
    src_win = window_partition(src, window_size_h=enc_win_h, window_size_w=enc_win_w)
    pos_embed_win = window_partition(pos_embed, window_size_h=enc_win_h, window_size_w=enc_win_w)
    mask_win = window_partition(mask.unsqueeze(1), window_size_h=enc_win_h, window_size_w=enc_win_w).squeeze(-1).permute(1,0)
    
    return src_win, pos_embed_win, mask_win


def enc_win_partition_reverse(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input for encoder
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1,0,2).view(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1).permute(0,3,1,2)
    return x


def window_partition(x, window_size_h, window_size_w):
    """
    window-rize input
    """
    B, C, H, W = x.shape
    x = x.permute(0,2,3,1)  # to (B, H, W, C)
    x = x.reshape(B, H // window_size_h, window_size_h, W // window_size_w, window_size_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size_h, window_size_w, C)
    windows = windows.reshape(-1, window_size_h*window_size_w, C).permute(1,0,2) # window_size_h * window_size_w, num_windows * Bs, C
    return windows

def query_partition(x, feat, win_h, win_w, test=False):
    bs, _, H, W = feat.shape
    x_scale = x.unsqueeze(1).repeat(1, bs, 1)
    nums_all, _, loc = x_scale.shape
    x_scale = x_scale.reshape(bs, H // win_h, win_h, W // win_w, win_w, loc)
    windows = x_scale.permute(0, 1, 3, 2, 4, 5).reshape(-1, win_h, win_w, loc)
    windows = windows.reshape(-1, win_h*win_w, loc).permute(1,0,2)
    return windows
    


def window_partition_reverse(windows, window_size_h, window_size_w, H, W):
    """
    reverse window-rized input
    """
    B = int(windows.shape[1] / (H * W / window_size_h / window_size_w))
    x = windows.permute(1,0,2).reshape(B, H // window_size_h, W // window_size_w, window_size_h, window_size_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    x = x.reshape(B, H*W, -1).permute(1,0,2)
    return x

