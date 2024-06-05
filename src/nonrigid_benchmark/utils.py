from functools import cache
from nonrigid_benchmark.io import load_sample, load_benchmark
from nonrigid_benchmark.ransac import nr_RANSAC
from tqdm import tqdm
import os, json
import cv2
import numpy as np

@cache
def getMatcher():
    return cv2.BFMatcher(crossCheck=True)

@cache
def getSIFT():
    return cv2.SIFT_create(nfeatures=2048)

def extract(sample, detector=None, descriptor=None, detectordescriptor=None):
    if detector is not None and descriptor is not None:
        keypoints = detector.detect(sample['image'], sample['mask'])
        keypoints, descriptors = descriptor.compute(sample['image'], keypoints)
    elif detectordescriptor is not None:
        keypoints, descriptors = detectordescriptor.detectAndCompute(sample['image'], sample['mask'])
    else:
        raise ValueError('Specify either detector and descriptor or detectordescriptor.')

    if len(keypoints) == 0:
        return [], []
    
    if isinstance(keypoints[0], cv2.KeyPoint):
        keypoints = [keypoint.pt for keypoint in keypoints]

    return keypoints, descriptors

def nr_matcher(features1, features2):
    keypoints1, descriptors1 = features1
    keypoints2, descriptors2 = features2

    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

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



def eval_loop(benchmark_path, predictions_path, match_fn, datasets = ['test_single_obj', 'test_multiple_obj', 'test_scale']):
    
    for dataset_type in datasets:
        selected_pairs = load_benchmark(os.path.join(benchmark_path, dataset_type))

        os.makedirs(os.path.join(predictions_path, dataset_type), exist_ok=True)
        
        for s_idx, split in enumerate(selected_pairs.keys()):
            json_file = os.path.join(predictions_path,dataset_type, f'{split}.json')
            if os.path.exists(json_file):
                print(f'File {json_file} already exists. Skipping...')
                continue
            
            predictions = []
            for pair_idx, pair in enumerate(tqdm(selected_pairs[split], desc=f'{dataset_type}/{split} ({s_idx+1}/{len(selected_pairs.keys())})')):
                sample1 = load_sample(pair[0])
                sample2 = load_sample(pair[1])

                keypoints1, keypoints2, matches = match_fn(sample1, sample2)
                
                predictions.append({
                    'keypoints1': keypoints1,
                    'keypoints2': keypoints2,
                    'matches': matches,
                })
                
            
            with open(json_file, 'w') as f:
                json.dump(predictions, f)

def eval_loop_cached(benchmark_path, predictions_path, extract_fn, match_fn, datasets = ['test_single_obj', 'test_multiple_obj', 'test_scale']):
    for dataset_type in datasets:
        selected_pairs = load_benchmark(os.path.join(benchmark_path, dataset_type))

        os.makedirs(os.path.join(predictions_path, dataset_type), exist_ok=True)
        print(f'Saving predictions to {os.path.join(predictions_path, dataset_type)}')
        
        # preextract all images
        unique_images = set()
        for s_idx, split in enumerate(selected_pairs.keys()):
            for pair in selected_pairs[split]:
                unique_images.add(pair[0])
                unique_images.add(pair[1])
        unique_images = list(unique_images)

        features = {}
        for image in tqdm(unique_images, desc=f'{dataset_type} | Extracting features'):
            features[image] = extract_fn(load_sample(image))

        for s_idx, split in enumerate(selected_pairs.keys()):
            json_file = os.path.join(predictions_path,dataset_type, f'{split}.json')
            
            predictions = []
            for pair_idx, pair in enumerate(tqdm(selected_pairs[split], desc=f'{dataset_type}/{split} ({s_idx+1}/{len(selected_pairs.keys())})')):
                keypoints1, keypoints2, matches = match_fn(features[pair[0]], features[pair[1]])
                
                predictions.append({
                    'keypoints1': keypoints1,
                    'keypoints2': keypoints2,
                    'matches': matches,
                })
            
                
            with open(json_file, 'w') as f:
                json.dump(predictions, f)

def match(sample1, sample2, detector=None, descriptor=None, detectordescriptor=None):
    
    if detector is not None and descriptor is not None:
        keypoints1 = detector.detect(sample1['image'], sample1['mask'])
        keypoints2 = detector.detect(sample2['image'], sample2['mask'])
        
        keypoints1, descriptors1 = descriptor.compute(sample1['image'], keypoints1)
        keypoints2, descriptors2 = descriptor.compute(sample2['image'], keypoints2)
    elif detectordescriptor is not None:
        keypoints1, descriptors1 = detectordescriptor.detectAndCompute(sample1['image'], sample1['mask'])
        keypoints2, descriptors2 = detectordescriptor.detectAndCompute(sample2['image'], sample2['mask'])
    else:
        raise ValueError('Specify either detector and descriptor or detectordescriptor.')

    # convert from opencv keypoints to tuples
    keypoints1 = [keypoint.pt for keypoint in keypoints1]
    keypoints2 = [keypoint.pt for keypoint in keypoints2]
    
    matcher = getMatcher()
    matches = matcher.match(descriptors1, descriptors2)

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