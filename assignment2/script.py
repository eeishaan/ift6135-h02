from models import GRU
import torch

def main():
    gru = GRU(16, 8, 4, 2, 26, 6, 0.2)
    allo = torch.tensor([[0, 11, 10, 11],[0, 10, 8, 6]])
    print(allo.shape)
    hidden = gru.init_hidden()
    output = gru(allo, hidden)
    print(output)
if __name__ == "__main__":
    main()
