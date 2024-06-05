from easy_local_features.feature.baseline_tfeat import TFeat_baseline as TFeat
from functools import partial
from ..utils import match, getSIFT, eval_loop

import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, help="Path to the model weights.")
    parser.add_argument("--model",   "-m", type=str, default=None, help="Path to the model weights.")
    parser.add_argument("--output",  "-o", type=str, default="./results/SIFT_TFeat_2048", help="Name of the model to save the predictions")
    return parser.parse_args()

if __name__ == "__main__":
    import torch
   
    args = parse()
    
    weights = args.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = TFeat(device=0)
    if weights is not None:
        print(f'Loading weights from {weights}')
        extractor.model.load_state_dict(torch.load(weights, map_location=device))
    
    match_fn = partial(match, detector=getSIFT(), descriptor=extractor)
    eval_loop(args.dataset, args.output, match_fn)
