from functools import partial
from ..utils import match, eval_loop_cached, nr_matcher, extract
import cv2
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, help="Path to the model weights.")
    parser.add_argument("--output",  "-o", type=str, default="./results/ORB_2048", help="Name of the model to save the predictions")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    
    # # ORB 2048
    extractor = cv2.ORB_create(nfeatures=2048)
    extract_fn = partial(extract, detectordescriptor=extractor)
    eval_loop_cached(args.dataset, args.output, extract_fn, nr_matcher)
