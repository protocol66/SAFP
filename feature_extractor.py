# Project 2 - 5/4/23
# Joshua Adams, Weston Beebe, Parth Patel, Jonathan Sanderson, Samuel Sylvester

from scipy import signal
import numpy as np

class FeatureExtractor:
    # extracts features from song data
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

    @staticmethod
    def method2(data):
        
        #get power spectral density (Welch's method)
        p_on_x = 8
        x = 1

        f, psd = signal.welch(data, fs=16000, window='blackman', nperseg=4096)
        peaks, properties = signal.find_peaks(psd, width=0, plateau_size=0, height=0)

        dim = (len(peaks),4)
        peaks_and_properties = np.empty(dim, dtype=np.float32) #[[f,width,plateau_size,height],...]
        for i in range(len(peaks)):
            peaks_and_properties[i][0] = f[peaks[i]]
            peaks_and_properties[i][1] = properties['widths'][i]
            peaks_and_properties[i][2] = properties['plateau_sizes'][i]
            peaks_and_properties[i][3] = properties['peak_heights'][i]

        s = sorted(peaks_and_properties, key=lambda p: p[3],reverse=True)
        
        features = np.empty(p_on_x*x, dtype=np.float32)
        for i in range(p_on_x):
            for j in range(x):
                features[i*x+j]=s[i][j]

        return features
        
        
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from dataloader import Dataset

    data, _, _ = Dataset(path='Project2data', split='train')[0]
    print(f'Shape of data: {data.shape}')

    features = FeatureExtractor.method1(data)
    print(f'Shape of features: {features.shape}')

    plt.figure()
    plt.plot(features)

    # plt.figure()
    # plt.specgram(data, Fs=16000, NFFT=2048, noverlap=1024)
    plt.show()
    
    # test method2
    features = FeatureExtractor.method2(data)
    print(f'Shape of features: {features.shape}')
    plt.plot(features)
    plt.show()