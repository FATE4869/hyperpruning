import data


class PBT_Dataloader():
    def __init__(self, data_path, batch_size, test_batch_size, bptt, device='cpu'):
        # args.data = '/home/ws8/caleb/dataset/PTB/penn/'
        self.data_path = data_path
        self.corpus = data.Corpus(data_path)
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.device = device
        self.bptt = bptt
        self.train_data = self.batchify(self.corpus.train, self.batch_size, self.device)
        self.val_data = self.batchify(self.corpus.valid, self.test_batch_size, self.device)
        self.test_data = self.batchify(self.corpus.test, self.test_batch_size, self.device)
        self.train_loader_len = int(self.train_data.size(0) / self.bptt)
        self.val_loader_len = int(self.val_data.size(0) / self.bptt)
        self.test_loader_len = int(self.test_data.size(0) / self.bptt)

        self.ntokens = len(self.corpus.dictionary)

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.
    def batchify(self, _data, bsz, device):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = _data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        _data = _data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        _data = _data.view(bsz, -1).t().contiguous()
        return _data.to(device)

    def get_batch(self, _data, i, bptt=None):
        # bptt_len = args.bptt
        if bptt:
            seq_len = min(bptt, len(_data) - 1 - i)
        else:
            seq_len = min(self.bptt, len(_data) - 1 - i)
        sample = _data[i:i+seq_len]
        target = _data[i+1:i+1+seq_len].view(-1)
        return sample, target


def main():
    data_path = '../dataset/PTB/penn/'

    dataloader = PBT_Dataloader(data_path, batch_size=32, test_batch_size=32, bptt=30)
    for i in range(2):
        sample, target = dataloader.get_batch(dataloader.train_data, i)
        print(sample, target)

if __name__ == '__main__':
    main()