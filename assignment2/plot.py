import argparse
from parse import *

import matplotlib.pyplot as plt
import numpy as np

def get_time(args):
    log_file = "./results/" + args.load_path + "/log.txt"
    file = open(log_file, "r")
    log = file.readlines()

    clock = [0.]

    for logs in log:
        logs = logs.replace("\t", " ")
        time = parse('epoch: {} train ppl: {} val ppl: {} best val: {} time (s) spent in epoch: {}', logs)[4]
        clock.append(float(time) + clock[-1])
    
    return clock

def save_lc_plot(args):
    LOAD_PATH = "./results/" + args.load_path + "/learning_curves.npy"

    learning_curves = np.load(LOAD_PATH)[()]
    train_ppls = learning_curves["train_ppls"]
    valid_ppls = learning_curves["val_ppls"]
    clock = get_time(args)

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
    plt.plot(valid_ppls[1:], label="Valid")
    plt.xlabel("Epochs")
    plt.ylabel("PPL")
    plt.legend()

    plt.subplot(122)
    plt.plot(clock[2:], train_ppls[1:], label="Train")
    plt.plot(clock[2:], valid_ppls[1:], label="Valid")
    plt.xlabel("Times")
    plt.ylabel("PPL")
    plt.legend()

    SAVE_PATH = "./images/" + args.save_path

    plt.savefig("{}{}.png".format(SAVE_PATH, title))

def plot_optimizer(args):
    if args.optimizer == "ADAM":
        LOAD_PATH_G = "./results/GRU_ADAM_model_GRU_optimizer_ADAM_initial_lr_0.0001_batch_size_20" \
                        +"_seq_len_35_hidden_size_1500_num_layers_2_dp_keep_prob_0.35_0/learning_curves.npy"
        LOAD_PATH_T = "./results/TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.002_batch_size=" \
                        + "64_seq_len=35_hidden_size=512_num_layers=3_dp_keep_prob=0.35_0/learning_curves.npy"
        LOAD_PATH_A = "./results/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0003_batch_size=32" \
                        + "_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_0/learning_curves.npy"
    if args.optimizer == "SGD":
        LOAD_PATH_A = "./results/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=0.0003_batch_size=32" \
                        + "_seq_len=35_hidden_size=1800_num_layers=2_dp_keep_prob=0.35_0/learning_curves.npy"
        LOAD_PATH_G = "./results/GRU_SGD_model=GRU_optimizer=SGD_initial_lr=12_batch_size=32" \
                        + "_seq_len=35_hidden_size=1800_num_layers=2_dp_keep_prob=0.35/learning_curves.npy"
        LOAD_PATH_T = "./results/TRANSFORMER_SGD_model=TRANSFORMER_optimizer=SGD_initial_lr=16" \
                        + "_batch_size=64_seq_len=35_hidden_size=512_num_layers=5_dp_keep_prob=0.9_0/learning_curves.npy"
    if args.optimizer == "SGDLS":
        LOAD_PATH_A = "./results/RNN_SGD_LR_SCHEDULE_model=RNN_optimizer=SGD_LR_SCHEDULE_initial_lr=2_batch_size=32" \
                        + "_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.35_0/learning_curves.npy"
        LOAD_PATH_G = "./results/GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=8_batch_size=32" \
                        + "_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35/learning_curves.npy"
        LOAD_PATH_T = "./results/TRANSFORMER_SGD_LR_SCHEDULE_model=TRANSFORMER_optimizer=SGD_LR_SCHEDULE_initial_lr=18" \
                        + "_batch_size=64_seq_len=35_hidden_size=512_num_layers=5_dp_keep_prob=0.9_0/learning_curves.npy"

    learning_curves_R = np.load(LOAD_PATH_A)[()]
    learning_curves_G = np.load(LOAD_PATH_G)[()]
    learning_curves_T = np.load(LOAD_PATH_T)[()]

    valid_ppls_R = learning_curves_R["val_ppls"]
    clock_R = learning_curves_R["clock"]

    valid_ppls_G = learning_curves_G["val_ppls"]
    clock_G = learning_curves_G["clock"]

    valid_ppls_T = learning_curves_T["val_ppls"]
    clock_T = learning_curves_T["clock"]

    plt.style.use('ggplot')     
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)

    plt.figure(figsize=(18, 6))
    plt.subplot(121)
    plt.plot(valid_ppls_R[1:], label="RNN")
    plt.plot(valid_ppls_G[1:], label="GRU")
    plt.plot(valid_ppls_T[1:], label="Transformer")
    plt.xlabel("Epochs")
    plt.ylabel("PPL")
    plt.legend()

    plt.subplot(122)
    plt.plot(clock_R[2:], valid_ppls_R[1:], label="RNN")
    plt.plot(clock_G[2:], valid_ppls_G[1:], label="GRU")
    plt.plot(clock_T[2:], valid_ppls_T[1:], label="Transformer")
    plt.xlabel("Times")
    plt.ylabel("PPL")
    plt.legend()

    SAVE_PATH = "./images/" + args.save_path

    plt.savefig("{}{}.png".format(SAVE_PATH, args.optimizer))

def plot_architecture(args):
    if args.archi == "GRU":
        LOAD_PATH_A = "./results/GRU_ADAM_model_GRU_optimizer_ADAM_initial_lr_0.0001_batch_size_20" \
                        +"_seq_len_35_hidden_size_1500_num_layers_2_dp_keep_prob_0.35_0/learning_curves.npy"
        LOAD_PATH_S = "./results/GRU_SGD_model=GRU_optimizer=SGD_initial_lr=12_batch_size=32" \
                        + "_seq_len=35_hidden_size=1800_num_layers=2_dp_keep_prob=0.35/learning_curves.npy"
        LOAD_PATH_SS = "./results/GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=8_batch_size=32" \
                        + "_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35/learning_curves.npy"
    if args.archi == "RNN":
        LOAD_PATH_A = "./results/RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0003_batch_size=32" \
                        + "_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_0/learning_curves.npy"
        LOAD_PATH_S = "./results/RNN_SGD_model=RNN_optimizer=SGD_initial_lr=0.0003_batch_size=32" \
                        + "_seq_len=35_hidden_size=1800_num_layers=2_dp_keep_prob=0.35_0/learning_curves.npy"
        LOAD_PATH_SS = "./results/RNN_SGD_LR_SCHEDULE_model=RNN_optimizer=SGD_LR_SCHEDULE_initial_lr=2_batch_size=32" \
                        + "_seq_len=35_hidden_size=512_num_layers=2_dp_keep_prob=0.35_0/learning_curves.npy"
    if args.archi == "Transformer":
        LOAD_PATH_A = "./results/TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.002_batch_size=" \
                        + "64_seq_len=35_hidden_size=512_num_layers=3_dp_keep_prob=0.35_0/learning_curves.npy"
        LOAD_PATH_S = "./results/TRANSFORMER_SGD_model=TRANSFORMER_optimizer=SGD_initial_lr=16" \
                        + "_batch_size=64_seq_len=35_hidden_size=512_num_layers=5_dp_keep_prob=0.9_0/learning_curves.npy"
        LOAD_PATH_SS = "./results/TRANSFORMER_SGD_LR_SCHEDULE_model=TRANSFORMER_optimizer=SGD_LR_SCHEDULE_initial_lr=18" \
                        + "_batch_size=64_seq_len=35_hidden_size=512_num_layers=5_dp_keep_prob=0.9_0/learning_curves.npy"

    learning_curves_A = np.load(LOAD_PATH_A)[()]
    learning_curves_S = np.load(LOAD_PATH_S)[()]
    learning_curves_SS = np.load(LOAD_PATH_SS)[()]

    valid_ppls_A = learning_curves_A["val_ppls"]
    clock_A = learning_curves_A["clock"]

    valid_ppls_S = learning_curves_S["val_ppls"]
    clock_S = learning_curves_S["clock"]

    valid_ppls_SS = learning_curves_SS["val_ppls"]
    clock_SS = learning_curves_SS["clock"]

    plt.style.use('ggplot')     
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=15)

    plt.figure(figsize=(18, 6))
    plt.subplot(121)
    plt.plot(valid_ppls_A[1:], label="ADAM")
    plt.plot(valid_ppls_S[1:], label="SGD")
    plt.plot(valid_ppls_SS[1:], label="SGD_LS")
    plt.xlabel("Epochs")
    plt.ylabel("PPL")
    plt.legend()

    plt.subplot(122)
    plt.plot(clock_A[2:], valid_ppls_A[1:], label="ADAM")
    plt.plot(clock_S[2:], valid_ppls_S[1:], label="SGD")
    plt.plot(clock_SS[2:], valid_ppls_SS[1:], label="SGD_LS")
    plt.xlabel("Times")
    plt.ylabel("PPL")
    plt.legend()

    SAVE_PATH = "./images/" + args.save_path

    plt.savefig("{}{}.png".format(SAVE_PATH, args.archi))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        type=str,
                        default="",
                        help="Full path where to load the learning curves array")
    parser.add_argument('--save_path',
                        type=str,
                        default="",
                        help="Path where to save the plots from /images/.")
    parser.add_argument('--optimizer',
                        type=str,
                        default="")
    parser.add_argument('--archi',
                        type=str,
                        default="")

    args = parser.parse_args()

    if args.load_path != "":
        save_lc_plot(args)

    if args.optimizer != "":
        plot_optimizer(args)

    if args.archi != "":
        plot_architecture(args)
    
    