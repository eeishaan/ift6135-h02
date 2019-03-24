import argparse
import os
import sys
import collections
import numpy
import torch
import torch.nn as nn

from models import GRU, RNN

np = numpy


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    
    This prevents Pytorch from trying to backpropagate into previous input 
    sequences when we use the final hidden states from one mini-batch as the 
    initial hidden states for the next mini-batch.
    
    Using the final hidden states in this way makes sense when the elements of 
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)

def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

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

def sequences_to_file(sequences, id_to_word):  
    if not os.path.exists("./generated_text"):
        os.mkdir("./generated_text")
    with open("./generated_text/testgru.txt", "w") as file:
        for i in range(sequences.shape[0]):
            sentence = ""
            for j in range(sequences.shape[1]):
                word = id_to_word[int(sequences[i, j])]
                sentence += word + " "
            file.write(sentence + "\n")

def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)

def run_epoch(model, data, is_train=False, lr=1.0):
    """
    One epoch of training/validation (depending on flag is_train).
    """
    if is_train:
        model.train()
    else:
        model.eval()
    epoch_size = ((len(data) // model.batch_size) - 1) // model.seq_len
    start_time = time.time()
    if args.model != 'TRANSFORMER':
        hidden = model.init_hidden()
        hidden = hidden.to(device)
    costs = 0.0
    iters = 0
    batch_iter = 0
    losses = []
    loss_t_tensor = torch.zeros(model.seq_len).cuda()
    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        hidden = model.init_hidden()
        hidden = hidden.to(device)
        batch_iter += 1
        if args.model == 'TRANSFORMER':
            batch = Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1,0)
            #print ("outputs.shape", outputs.shape)
        else:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            outputs, hidden = model(inputs, hidden)

        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
        tt = torch.squeeze(targets.view(-1, model.batch_size * model.seq_len))

        # LOSS COMPUTATION
        # This line currently averages across all the sequences in a mini-batch 
        # and all time-steps of the sequences.
        # For problem 5.3, you will (instead) need to compute the average loss 
        #at each time-step separately. 
        loss = loss_fn(outputs.contiguous().view(-1, model.vocab_size), tt)
        loss_t = []
        for i in range(targets.shape[0]):
            loss_t.append(loss_fn(outputs[i, :, :], targets[i, :]))
        loss_t_tensor += torch.stack(loss_t).detach()
        costs += loss.data.item() * model.seq_len
        losses.append(costs)
        iters += model.seq_len
        if args.debug:
            print(step, loss)
        if is_train:  # Only update parameters if training 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            if args.optimizer == 'ADAM':
                optimizer.step()
            else: 
                for p in model.parameters():
                    if p.grad is not None:
                        p.data.add_(-lr, p.grad.data)
            if step % (epoch_size // 10) == 10:
                print('step: '+ str(step) + '\t' \
                    + 'loss: '+ str(costs) + '\t' \
                    + 'speed (wps):' + str(iters * model.batch_size / (time.time() - start_time)))
    loss_t_tensor = loss_t_tensor / batch_iter
    return np.exp(costs / iters), losses

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
        model = GRU(**model_args).cuda()
    if model_type == "RNN":
        model = RNN(**model_args).cuda()
    if model_type == "TRANSFORMER":
        pass
    model.load_state_dict(torch.load(model_path + "best_params.pt"))
    word_to_id, id_to_word = build_vocab("./data/ptb.train.txt")
    random_batch = torch.tensor(
        [np.random.randint(0, len(id_to_word)) for i in range(gen_length)]
    )
    sequences = model.generate(random_batch, 0, gen_length)
    sequences_to_file(sequences, id_to_word)

    valid_path = "./data/ptb.valid.txt"
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    a, b = run_epoch(model, valid_data, 1)