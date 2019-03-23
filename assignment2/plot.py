import argparse

import matplotlib.pyplot as plt
import numpy as np

def save_lc_plot(args):
    learning_curves = np.load(args.load_path)[()]
    train_ppls = learning_curves["train_ppls"]
    valid_ppls = learning_curves["val_ppls"]

    if args.load_path[2] == "G":
        title = "GRU"
    if args.load_path[2] == "R":
        title = "RNN"
    if args.load_path[2] == "T":
        title = "Transformer"

    plt.style.use('ggplot')     
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)

    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.plot(train_ppls[1:], label="Train")
    plt.plot(valid_ppls[1:], label="Valid")
    plt.xlabel("Epochs")
    plt.ylabel("PPL")
    plt.legend()

    plt.savefig("{}{}.png".format(args.save_path, title))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        type=str,
                        default='./GRU_SGD_LR_SCHEDULE_num_epochs_2_0/learning_curves.npy',
                        help="Full path where to load the learning curves array")
    parser.add_argument('--save_path',
                        type=str,
                        default="./images/",
                        help="Path where to save the plots.")
    args = parser.parse_args()

    save_lc_plot(args)