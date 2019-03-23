import argparse
import os
import sys
import collections
import numpy
import torch
import torch.nn as nn

from models import GRU, RNN

np = numpy


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


def load_args(path):
    config = np.genfromtxt(path + "exp_config.txt", dtype=str)
    args = {
        "emb_size": None,
        "hidden_size": None,
        "seq_len": None,
        "batch_size": None,
        "vocab_size": 10000,
        "num_layers": None,
        "dp_keep_prob": None,
    }
    for arg, value in config:
        if arg in args:
            try:
                args[arg] = int(value)
            except:
                try:
                    args[arg] = float(value)
                except:
                    args[arg] = value
    return args


def read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation")
    parser.add_argument("--model", type=str, default="GRU")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr"
        + "=10_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2"
        + "_dp_keep_prob=0.35_save_best_12/",
    )
    parser.add_argument("--gen_length", type=int, default=20)
    parser.add_argument("--num_seq", type=int, default=10)
    args = parser.parse_args()
    model_type = args.model
    model_path = args.model_path
    gen_length = args.gen_length
    num_seq = args.num_seq

    model_args = load_args(args.model_path)

    if model_type == "GRU":
        model = GRU(**model_args)
    if model_type == "RNN":
        model = RNN(**model_args)

    model.load_state_dict(torch.load(model_path + "best_params.pt", map_location="cpu"))
    _, id_to_word = build_vocab("./data/ptb.train.txt")
    random_batch = torch.tensor(np.random.rand(0, len(id_to_word)))
    sequences = model.generate(random_batch, 0, gen_length)
