import os
import json
from .io import load_sample, load_benchmark
from .compute import warp_keypoints
import argparse
from tqdm import tqdm
import numpy as np
import multiprocessing
from scipy.spatial.distance import cdist

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the matches file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--split", type=str, required=True, help="Split to evaluate")
    parser.add_argument("--matching_th", type=float, default=3, help="Matching threshold")
    parser.add_argument("--nproc", type=int, default=1, help="Parallel processors")
    parser.add_argument("--plot", action='store_true', help="Plot the results")
    return parser.parse_args()

def plot(sample1, sample2, keypoints1, keypoints2, mkpts1, mkpts2):
    from nonrigid_benchmark.visualize.utils import plot_pair, plot_matches, plot_keypoints, save
    
    plot_pair(sample1['image'], sample2['image'], title='Images')
    plot_keypoints(keypoints1, keypoints2)
    plot_matches(mkpts1, mkpts2)
    save('pair.png')

def eval_pair(args):
    pair, prediction, matching_th, make_plot = args
    sample1 = load_sample(pair[0], read_coords=True, read_segmentation=True)
    sample2 = load_sample(pair[1], read_coords=True, read_segmentation=True)
    
    matches = np.array(prediction['matches'])
    keypoints1 = np.array(prediction['keypoints1'])
    keypoints2 = np.array(prediction['keypoints2'])
    
    keypoints_mask1 = [sample1['mask'][int(kp[1]), int(kp[0])] > 0 for kp in keypoints1]
    keypoints_mask2 = [sample2['mask'][int(kp[1]), int(kp[0])] > 0 for kp in keypoints2]

    matches = np.array([match for match in matches if keypoints_mask1[match[0]] and keypoints_mask2[match[1]]])

    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return {
            'ms':0,
            'ma':0,
            'rr':0,
        }

    # masked_keypoints1 = [kp for kp, mask in zip(keypoints1, keypoints_mask1) if mask]
    # masked_keypoints2 = [kp for kp, mask in zip(keypoints2, keypoints_mask2) if mask]

    projection_1to2 = warp_keypoints(
        keypoints1=keypoints1,
        coords1=sample1['uv_coords'],
        coords2=sample2['uv_coords'],
        segmentation1=sample1['segmentation'],
        segmentation2=sample2['segmentation'],
        threshold=300,
    )
    
    keypoints_1to2 = projection_1to2['keypoints']
    valid_count_1to2 = sum(keypoints_1to2[keypoints_mask1,0] != -1)
    # import pdb; pdb.set_trace()
    if valid_count_1to2 > 0:
        repeatablility_1to2 = sum(cdist(keypoints2[keypoints_mask2], keypoints_1to2[keypoints_mask1]).min(axis=1) < matching_th) / valid_count_1to2
    else:
        repeatablility_1to2 = 0
    
    minimun_keypoints = min(sum(keypoints_mask1), sum(keypoints_mask2))
    
    if len(matches) == 0:
        return {
            'ms':0,
            'ma':0,
            'rr':repeatablility_1to2,
        }
        
    mkpts1 = keypoints1[matches[:, 0]]
    mkpts2 = keypoints2[matches[:, 1]]

    if make_plot:
        plot(sample1, sample2, keypoints1, keypoints2, mkpts1, mkpts2)
    
    gt_mkpts2 = keypoints_1to2[matches[:, 0]]
    dists = np.linalg.norm(mkpts2 - gt_mkpts2, axis=1)        

    ms = (dists < matching_th).sum() / minimun_keypoints
    ma = (dists < matching_th).sum() / len(matches)
    
    return {
        'ms':ms,
        'ma':ma,
        'rr':repeatablility_1to2,
    }


def main():
    args = parse()    
    selected_pairs = load_benchmark(args.dataset)
    nproc = args.nproc
    
    predictions = json.load(open(args.input, 'r'))
    
    outfile_path = args.output
    split = args.split
    
    metrics = {
        'ms': [],
        'ma': [],
        'rr': [],
    }

    if nproc > 1:    
        with multiprocessing.Pool(nproc) as pool:
            args = [(pair, prediction, args.matching_th, args.plot) for pair, prediction in zip(selected_pairs[split], predictions)]
            results = list(tqdm(pool.imap(eval_pair, args), total=len(args)))
            for result in results:
                metrics['ms'].append(result['ms'])
                metrics['ma'].append(result['ma'])
                metrics['rr'].append(result['rr'])
    else:
        for pair, prediction in zip(selected_pairs[split], predictions):
            result = eval_pair((pair, prediction, args.matching_th, args.plot))
            metrics['ms'].append(result['ms'])
            metrics['ma'].append(result['ma'])
            metrics['rr'].append(result['rr'])
        
    # mean score
    ms = np.mean(metrics['ms'])
    ma = np.mean(metrics['ma'])
    rr = np.mean(metrics['rr'])
    
    with open(outfile_path, 'w') as f:
        f.write(f"{ms},{ma},{rr}")
        
        
if __name__ == "__main__":
    main()