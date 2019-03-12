from models import GRU

def main():
    gru = GRU(5, 50, 50, 1, 26, 2, 0.2)
    print(gru.parameters())
    print(gru._weights)
    for layer in gru._weights:
        print(layer)
        for name in layer:
            print(name[0]=="b")
            if name[0] == "w" or "u":
                getattr(gru, name).data.uniform_(-1, 1)
            if name[0] == "b":
                getattr(gru, name).data.zero_()
                print(getattr(gru, name))

if __name__ == "__main__":
    main()
