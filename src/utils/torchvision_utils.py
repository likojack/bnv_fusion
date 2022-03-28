import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_


def visualize_prob(prob, cmap=cv2.COLORMAP_BONE):
    """
    prob: (H, W) 0~1
    """
    x = (255*prob).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_


def depth_visualizer(data, min_depth, max_depth):
    """
    Args:
        data (HxW): depth data
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """

    mask = np.logical_and(data > min_depth, data < max_depth)
    inv_depth = 1 / (data + 1e-6)
    vmax = np.percentile(1/(data[mask]+1e-6), 90)
    normalizer = mpl.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    vis_data = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
    return vis_data