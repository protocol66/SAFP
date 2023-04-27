from scipy import signal
import numpy as np

class FeatureExtractor:
    @staticmethod
    def method1(data):
        
        # get spectrogram
        f, t, sgram = signal.stft(data, nperseg=2048, fs=16000)
        sgram_max = np.argmax(np.abs(sgram), axis=0)
        
        features = np.empty(t.shape, dtype=np.float32)
        t = np.array(t*16000, dtype=np.int32)
        f = np.array(f, dtype=np.float32)
        
        for i in range(0, len(t)-1):
            features[i] = f[sgram_max[i]]
            
        return features
        
        
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from dataloader import Dataset

    data, _, _ = Dataset(path='project_final/Project2data', split='train')[0]
    print(f'Shape of data: {data.shape}')

    features = FeatureExtractor.method1(data)

    plt.figure()
    plt.plot(features)

    plt.figure()
    plt.specgram(data, Fs=16000, NFFT=2048, noverlap=1024)

    plt.show()