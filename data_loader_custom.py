from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os, pdb, pickle, random, math
       
from multiprocessing import Process, Manager   


class pathSpecDataset(Dataset):
    """Dataset class for using a path to spec folders,
    path for labels,
    generates random windowed subspec examples,
    associated labels,
    optional conditioning."""
    def __init__(self, config, spmel_params):
        """Initialize and preprocess the dataset."""
        self.config = config
        melsteps_per_second = spmel_params['sr'] / spmel_params['hop_size']
        self.window_size = math.ceil(config.chunk_seconds * melsteps_per_second) * config.chunk_num
        dir_name, _, fileList = next(os.walk(self.config.spmel_dir))
        fileList = sorted(fileList)
        dataset = []
        for file_name in fileList:
            if file_name.endswith('.npy'):
                spmel = np.load(os.path.join(dir_name, file_name))
                dataset.append(spmel)
        self.dataset = dataset
        self.num_specs = len(dataset)
        
    """__getitem__ selects a speaker and chooses a random subset of data (in this case
    an utterance) and randomly crops that data. It also selects the corresponding speaker
    embedding and loads that up. It will now also get corresponding pitch contour for such a file"""

    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.dataset
        # spkr_data is literally a list of skpr_id, emb, and utterances from a single speaker
        spmel = dataset[index]
        # pick random spmel_chunk with random crop
        """Ensure all spmels are the length of (self.window_size * chunk_num)"""
        if spmel.shape[0] >= self.window_size:
            difference = spmel.shape[0] - self.window_size
            offset = random.randint(0, difference)
        adjusted_length_spmel = spmel[offset : offset + self.window_size]
        # may need to set chunk_num to constant value so that all tensor sizes are of known shape for the LSTM
        # a constant will also mean it is easier to group off to be part of the same recording
        # the smallest is 301 frames. If the window sizes are 44, then that 6 full windows each
        return adjusted_length_spmel

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_specs
