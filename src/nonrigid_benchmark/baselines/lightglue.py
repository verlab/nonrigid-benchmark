from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
from easy_local_features.matching.baseline_lightglue import LightGlue_baseline
import numpy as np

from functools import partial
from ..utils import match, eval_loop_cached, nr_matcher, extract
from ..ransac import nr_RANSAC
import cv2
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, help="Path to the model weights.")
    parser.add_argument("--output",  "-o", type=str, default="./results/SuperPoint_Lightglue_2048", help="Name of the model to save the predictions")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    
    extractor = SuperPoint_baseline(max_keypoints=2048, device=0)
    lightglue = LightGlue_baseline()
    extract_fn = partial(extract, detectordescriptor=extractor)

    def matcher_fn(features1, features2):
        keypoints1, descriptors1 = features1
        keypoints2, descriptors2 = features2
        
        _, _, matches = lightglue.match(np.array(keypoints1), np.array(keypoints2), descriptors1, descriptors2)

        if len(matches) == 0:
            return keypoints1, keypoints2, []
        
        # extract the keypoints from the matches for the nr_ransac
        src_pts = np.float32([keypoints1[m.queryIdx] for m in matches])
        tgt_pts = np.float32([keypoints2[m.trainIdx] for m in matches])
        
        #Computes non-rigid RANSAC
        try:
            inliers = nr_RANSAC(src_pts, tgt_pts, device='cuda', thr = 0.2)
        except:
            inliers = np.ones(len(matches), dtype=bool)
        
        good_matches = [matches[i] for i in range(len(matches)) if inliers[i]]
        matches = [[match.queryIdx, match.trainIdx] for match in good_matches]
        
        return keypoints1, keypoints2, matches

    eval_loop_cached(args.dataset, args.output, extract_fn, matcher_fn)

