import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

import torch

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      ax, scores=None, colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[0] == class_ids.shape[0]

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    ax.axis('off')

    # masked_image = image.type(torch.IntTensor).clone()
    masked_image = image.cpu().clone()
    masked_image = masked_image.numpy()
    masked_image = np.transpose(masked_image, (1,2,0))
    for i in range(N):
        color = colors[i]

        boxes[i]=boxes[i].type(torch.IntTensor)
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((y1, x1), abs(y2-y1), abs(x2-x1), linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor="red", facecolor='none')
        ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1-11, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[i,:, :]
        mask=mask.cpu()
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color, alpha=0.9)
            ax.add_patch(p)
    ax.imshow((masked_image * 255).astype(np.uint8))
    

def display_groundtruth_vs_pred(image,original,pred,figsize=(16, 9)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Ground Truth vs Prediction",y=0.9,size="large",weight="bold")
    # Original
    display_instances(image, original["boxes"], original["masks"], original["class_ids"], original["class_names"], ax1)

    # Pred
    display_instances(image, pred["boxes"], pred["masks"], pred["class_ids"], pred["class_names"], ax2)
    plt.show()