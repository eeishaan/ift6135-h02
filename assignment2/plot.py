import argparse

import matplotlib.pyplot as plt
import numpy as np

def save_lc_plot(args):
    LOAD_PATH = "./results/" + args.load_path + "/learning_curves.npy"

    learning_curves = np.load(LOAD_PATH)[()]
    train_ppls = learning_curves["train_ppls"]
    valid_ppls = learning_curves["val_ppls"]
    clock = learning_curves["clock"]

    if args.load_path[0] == "G":
        title = "GRU"
    if args.load_path[0] == "R":
        title = "RNN"
    if args.load_path[0] == "T":
        title = "Transformer"

    plt.style.use('ggplot')     
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)

    plt.figure(figsize=(18, 6))
    plt.subplot(121)
    plt.plot(train_ppls[1:], label="Train")
    plt.plot(val_ppls[1:], label="Valid")
    plt.xlabel("Epochs")
    plt.ylabel("PPL")
    plt.legend()

    plt.subplot(122)
    plt.plot(clock[1:], train_ppls[1:], label="Train")
    plt.plot(clock[1:], val_ppls[1:], label="Valid")
    plt.xlabel("Times")
    plt.ylabel("PPL")
    plt.legend()

    SAVE_PATH = "./images/" + args.save_path

    plt.savefig("{}{}.png".format(SAVE_PATH, title))

def plot_optimizer(args):
    if args.optimizer == "ADAM":
        LOAD_PATH_R = "./results/"
        LOAD_PATH_G = "./results/"
        LOAD_PATH_T = "./results/"
    if args.optimizer == "SGD":
        LOAD_PATH_R = "./results/"
        LOAD_PATH_G = "./results/"
        LOAD_PATH_T = "./results/"
    if args.optimizer == "SGDLS":
        LOAD_PATH_R = "./results/"
        LOAD_PATH_G = "./results/"
        LOAD_PATH_T = "./results/"

    learning_curves_R = np.load(LOAD_PATH_R)[()]
    learning_curves_G = np.load(LOAD_PATH_G)[()]
    learning_curves_T = np.load(LOAD_PATH_T)[()]

    train_ppls_R = learning_curves_R["train_ppls"]
    valid_ppls_R = learning_curves_R["val_ppls"]
    clock_R = learning_curves_R["clock"]

    train_ppls_G = learning_curves_G["train_ppls"]
    valid_ppls_G = learning_curves_G["val_ppls"]
    clock_G = learning_curves_G["clock"]

    train_ppls_T = learning_curves_T["train_ppls"]
    valid_ppls_T = learning_curves_T["val_ppls"]
    clock_T = learning_curves_T["clock"]

    plt.style.use('ggplot')     
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)

    plt.figure(figsize=(18, 6))
    plt.subplot(121)
    plt.title()
    plt.plot(val_ppls_R[1:], label="RNN")
    plt.plot(val_ppls_G[1:], label="GRU")
    plt.plot(val_ppls_T[1:], label="Transformer")
    plt.xlabel("Epochs")
    plt.ylabel("PPL")
    plt.legend()

    plt.subplot(122)
    plt.plot(clock_R[1:], val_ppls_R[1:], label="RNN")
    plt.plot(clock_G[1:], val_ppls_G[1:], label="GRU")
    plt.plot(clock_T[1:], val_ppls_T[1:], label="Transformer")
    plt.xlabel("Times")
    plt.ylabel("PPL")
    plt.legend()

    SAVE_PATH = "./images/" + args.save_path

    plt.savefig("{}{}.png".format(SAVE_PATH, args.optimizer))

def plot_architecture(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        type=str,
                        default='GRU_SGD_LR_SCHEDULE_num_epochs_2_0',
                        help="Full path where to load the learning curves array")
    parser.add_argument('--save_path',
                        type=str,
                        default="",
                        help="Path where to save the plots.")
    args = parser.parse_args()

    save_lc_plot(args)