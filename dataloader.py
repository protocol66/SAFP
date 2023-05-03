from scipy.io import wavfile
from pathlib import Path
import numpy as np

from typing import Union


class Dataset():
    def __init__(self, path='Project2data', split='train', transform=None):
        
        self.path = Path(path).joinpath(split)
        self.transform = transform if transform else lambda x: x
            
        # make sure the path exists
        if not self.path.exists():
            raise Exception(f'Path {self.path} does not exist')
        
        # make sure the path is a directory
        if not self.path.is_dir():
            raise Exception(f'Path {self.path} is not a directory')
        
        # get a list of all the files in the directory
        self._dataset = list(self.path.glob('*.wav'))
        
        # make list is not empty
        if len(self._dataset) == 0:
            raise Exception(f'No files found in {self.path}')
        
        #sort the list
        self._dataset = sorted(self._dataset, key=lambda x: x.name)
        
    def __len__(self):
        return len(self._dataset)
    
    def _load_file(self, filename:str):
        return wavfile.read(filename)
    
    def _get_label(self, path:Path):
        return int(path.stem.split('_')[0])
    
    def __getitem__(self, idx: Union[int, list, slice]):
        data = []
        labels = []
        sampling_rates = []
        
        if isinstance(idx, slice):            
            for i in range(*idx.indices(len(self))):
                filename = self._dataset[i]
                fs, d = self.transform(self._load_file(filename))
                data.append(d)
                labels.append(self._get_label(filename))
                sampling_rates.append(fs)
                
            return np.array(data), np.array(labels), np.array(sampling_rates)
            
        elif isinstance(idx, list) or isinstance(idx, np.ndarray):
            for i in idx:
                filename = self._dataset[i]
                fs, d = self.transform(self._load_file(filename))
                data.append(d)
                labels.append(self._get_label(filename))
                sampling_rates.append(fs)
                
            return np.array(data), np.array(labels), np.array(sampling_rates)
                
        elif isinstance(idx, int):
            filename = self._dataset[idx]
            fs, d = self.transform(self._load_file(filename))
            data = d
            labels = self._get_label(filename)
            sampling_rates = fs
            
            return data, labels, sampling_rates
            
        else:
            raise TypeError(f'Invalid argument type: {type(idx)}')

        
        

class Dataloader():
    def __init__(self, dataset:Dataset, batch_size:int=1, shuffle:bool=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        
        if self.shuffle:
            self.idxs = np.random.permutation(len(self.dataset))
        else:
            self.idxs = np.arange(len(self.dataset))
            
        # accumulate data and labels
        for i in range(0, len(self.dataset), self.batch_size):
            data, labels, sample_rates = self.dataset[self.idxs[i:i+self.batch_size]]
            yield data, labels, sample_rates



if __name__ == '__main__':

    ds = Dataset('./Project2data', split='train')
    dl = Dataloader(ds, batch_size=1, shuffle=True)

    count = 0
    for data, labels, sample_rate in dl:
        print(f'Fs: {sample_rate[0]}, Length: {len(data[0])}, Label: {labels[0]}')
        assert len(data[0]) == 80000, 'Length of data is not 80000'
        assert sample_rate[0] == 16000, 'Sampling rate is not 16000'
        count += 1
        
    print(f'Number of Samples: {count}')
    print(f"First sample: {data[0].min()}, {data[0].max()}")