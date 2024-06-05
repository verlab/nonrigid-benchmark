from easy_local_features.feature.baseline_deal import DEAL_baseline as DEAL
from functools import partial
from ..utils import match, getSIFT, eval_loop, eval_loop_cached, extract, nr_matcher

import argparse

DATASET_DEFAULT = '/work/cadar/Datasets/nonrigid-simulation-benchmark/'
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, help="Path to the model weights.")
    parser.add_argument("--model",   "-m", type=str, default=None, help="Path to the model weights.")
    parser.add_argument("--output",  "-o", type=str, default="./results/SIFT_DEAL_2048", help="Name of the model to save the predictions")
    return parser.parse_args()

if __name__ == "__main__":
    import torch
   
    args = parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = DEAL(device_id=0)
    weights = args.model
    if weights is not None:
        print(f'Loading weights from {weights}')
        extractor.deal.net.load_state_dict(torch.load(weights, map_location=device))
    
    extract_fn = partial(extract, detector=getSIFT(), descriptor=extractor)
    eval_loop_cached(args.dataset, args.output, extract_fn, nr_matcher)
