import torch
import numpy as np

class CategoriesSampler():

    def __init__(self, label, n_per=9):
        self.n_per = n_per
        self.n_batch = label.shape[0] // n_per // 60

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def shuffle_ind(self):
        for i in range(len(self.m_ind)):
            self.m_ind[i] = self.m_ind[i][torch.randperm(self.m_ind[i].shape[0])]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        self.shuffle_ind()
        for i_batch in range(self.n_batch):
            batch = []
            for c in range(len(self.m_ind)):
                l = self.m_ind[c]  # all data indexs of this class
                batch.append(l[i_batch*self.n_per:(i_batch+1)*self.n_per])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

