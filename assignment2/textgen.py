import argparse
import os
import sys

import numpy
import torch
import torch.nn as nn

from models import GRU, RNN

np = numpy

def load_args(path):
    config = np.load(path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation")
    parser.add_argument(
        "--model", type=str, default="RNN"
    )
    parser.add_argument(
        "--model_path",
        type=str,
    )

