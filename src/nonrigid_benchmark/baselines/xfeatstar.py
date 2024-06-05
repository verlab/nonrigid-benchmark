import numpy as np
from functools import partial
from ..utils import match, eval_loop_cached, nr_matcher, extract, eval_loop
from ..ransac import nr_RANSAC
import cv2
import argparse
import torch


class warpXFeat():
    def __init__(self):
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True)
        
    def match(self, sample1, sample2):
        mask0 = sample1['mask']
        mask1 = sample2['mask']
        image1 = sample1['image']
        image2 = sample2['image']
        image1 = image1 * (mask0[:, :, None] > 0)
        image2 = image2 * (mask1[:, :, None] > 0)
        
        mkpts0, mkpts1 = self.xfeat.match_xfeat_star(image1, image2)
        
        matches = [[i,i] for i in range(len(mkpts0))]
        return mkpts0.tolist(), mkpts1.tolist(), matches
        

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, help="Path to the model weights.")
    parser.add_argument("--output",  "-o", type=str, default="./results/XFeatStar", help="Name of the model to save the predictions")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()

    extractor = warpXFeat()
    eval_loop(args.dataset, args.output, extractor.match)
