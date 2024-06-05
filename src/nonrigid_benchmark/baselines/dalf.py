from easy_local_features.feature.baseline_dalf import DALF_baseline as DALF
from nonrigid_benchmark.ransac import nr_RANSAC

import cv2
from functools import cache
import numpy as np
import argparse

@cache
def getMatcher():
    return cv2.BFMatcher(crossCheck=True)

def match(extractor:DALF, sample1, sample2):
    keypoints1, descriptors1 = extractor.detectAndCompute(sample1['image'], top_k = 2048)
    keypoints2, descriptors2 = extractor.detectAndCompute(sample2['image'], top_k = 2048)
    
    # convert from opencv keypoints to tuples
    keypoints1 = [keypoint.pt for keypoint in keypoints1]
    keypoints2 = [keypoint.pt for keypoint in keypoints2]
    
    matcher = getMatcher()
    matches = matcher.match(descriptors1, descriptors2)

    # extract the keypoints from the matches for the nr_ransac
    src_pts = np.float32([keypoints1[m.queryIdx] for m in matches])
    tgt_pts = np.float32([keypoints2[m.trainIdx] for m in matches])
    
    #Computes non-rigid RANSAC
    inliers = nr_RANSAC(src_pts, tgt_pts, device='cuda', thr = 0.2)
    
    good_matches = [matches[i] for i in range(len(matches)) if inliers[i]]
    matches = [[match.queryIdx, match.trainIdx] for match in good_matches]
    
    return keypoints1, keypoints2, matches

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, help="Path to the model weights.")
    parser.add_argument("--model",  "-m", type=str, default=None, help="Path to the model weights.")
    return parser.parse_args()

if __name__ == "__main__":
    from nonrigid_benchmark.io import load_sample, load_benchmark
    from tqdm import tqdm
    import os, json
    import torch
   
    args = parse()
    
    weights = args.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = DALF(weights, device)

    #Match using vanilla opencv matcher
    matcher = cv2.BFMatcher(crossCheck = True)
    
    # benchmark_path = './assets/sample_dataset'
    datasets = ['test_single_obj', 'test_multiple_obj', 'test_scale']
    for dataset_type in datasets:

        benchmark_path = f'{args.dataset}/{dataset_type}'
        selected_pairs = load_benchmark(benchmark_path)

        predictions_path = f'./results/DALF_2048_finetune/{dataset_type}'
        os.makedirs(predictions_path, exist_ok=True)

        for s_idx, split in enumerate(selected_pairs.keys()):
        
            predictions = []
            for pair_idx, pair in enumerate(tqdm(selected_pairs[split], desc=f'{dataset_type}/{split} ({s_idx+1}/{len(selected_pairs.keys())})')):
                sample1 = load_sample(pair[0])
                sample2 = load_sample(pair[1])
                
                keypoints1, keypoints2, matches = match(extractor, sample1, sample2)
                
                predictions.append({
                    'keypoints1': keypoints1,
                    'keypoints2': keypoints2,
                    'matches': matches,
                })
                
            json_file = os.path.join(predictions_path, f'{split}.json')
            with open(json_file, 'w') as f:
                json.dump(predictions, f)