import cv2
import os
import json

def load_sample(rgb_path:str, read_coords:bool=False, read_segmentation:bool=False):
    image = cv2.imread(rgb_path)
    mask = cv2.imread(rgb_path.replace('rgba', 'bgmask'), cv2.IMREAD_UNCHANGED)

    sample = {
        'image': image,
        'mask': mask,
    }

    if read_coords:
        sample['uv_coords'] = cv2.imread(rgb_path.replace('rgba', 'uv'), cv2.IMREAD_UNCHANGED)

    if read_segmentation:
        sample['segmentation'] = cv2.imread(rgb_path.replace('rgba', 'segmentation'), cv2.IMREAD_UNCHANGED)
        
    return sample


def load_benchmark(benchmark_path:str):
    assert os.path.isdir(benchmark_path), f"Path {benchmark_path} is not a directory"
    assert os.path.isfile(os.path.join(benchmark_path, 'selected_pairs.json')), f"File selected_pairs.json not found in {benchmark_path}"
    
    with open(os.path.join(benchmark_path, 'selected_pairs.json'), 'r') as f:
        selected_pairs = json.load(f)
        
    for split in selected_pairs:
        for idx, pair in enumerate(selected_pairs[split]):
            selected_pairs[split][idx] = [
                os.path.join(benchmark_path, pair[0]),
                os.path.join(benchmark_path, pair[1])
            ]
        
    return selected_pairs
