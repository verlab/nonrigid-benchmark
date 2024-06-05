from functools import partial
from ..utils import match, eval_loop_cached, nr_matcher, extract
import cv2
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, help="Path to the model weights.")
    parser.add_argument("--output",  "-o", type=str, default="./results/SIFT_FREAK_2048", help="Name of the model to save the predictions")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    
    detector = cv2.SIFT_create(nfeatures=2048)
    descriptor = cv2.xfeatures2d.FREAK_create()

    extract_fn = partial(extract, detector=detector, descriptor=descriptor)
    eval_loop_cached(args.dataset, args.output, extract_fn, nr_matcher)
