import numpy as np
from functools import partial
from ..utils import match, eval_loop_cached, nr_matcher, extract
from ..ransac import nr_RANSAC
import cv2
import argparse
import torch


class warpXFeat():
    def __init__(self):
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 2084)
        
    def detectAndCompute(self, img, mask=None):
        torch_img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to('cuda')
        output = self.xfeat.detectAndCompute(torch_img, top_k = 2048)[0]
        keypoints = output['keypoints'].cpu().numpy()
        descriptors = output['descriptors'].cpu().numpy()
    
        return keypoints, descriptors        
        
    def match(self, feats1, feats2):
        keypoints1, descriptors1 = feats1
        keypoints2, descriptors2 = feats2
        
        desc1 = torch.from_numpy(descriptors1).to('cuda').float()
        desc2 = torch.from_numpy(descriptors2).to('cuda').float()
        m0, m1 = self.xfeat.match(desc1, desc2)
        
        matches = [[m0[i].item(), m1[i].item()] for i in range(len(m0))]
        return keypoints1.tolist(), keypoints2.tolist(), matches
        

DATASET_DEFAULT = '/work/cadar/Datasets/nonrigid-simulation-benchmark/'
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, help="Path to the model weights.")
    parser.add_argument("--output",  "-o", type=str, default="./results/XFeat_2048", help="Name of the model to save the predictions")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()

    extractor = warpXFeat()

    extract_fn = partial(extract, detectordescriptor=extractor)


    eval_loop_cached(args.dataset, args.output, extract_fn, extractor.match)
