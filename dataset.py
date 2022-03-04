from torch.utils import data
from torch.utils.data.dataset import Dataset
import pandas as pd
import torch
from glob import iglob
import torchaudio
import torchaudio.functional as audioF
import pandas as pd

class BUTRIR(Dataset):
    def __init__(self, transform=None):
        data_root = '/home/shanxr/project/rirData/'
        self.wav = []
        self.transform = transform
        # load audio
        for file in iglob(data_root +'RIR'+'/*.wav'):
            self.wav.append(file)
        # load label
        self.df = pd.read_csv(data_root + 'rir_acoustic_param.csv', encoding='utf-8', index_col=0)
#        print(df.index)
#        print(df.loc['Hotel_SkalskyDvur_ConferenceRoom2-MicID01-SpkID01_20170906_S-19-RIR-IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav'])
        
        pass

    def __len__(self):
        return len(self.wav)
        pass

    def __normalize(self, t):
        # Subtract the mean, and scale to the interval [-1,1]
        return 2.0 * (t - t.min()) / (t.max() - t.min()) - 1.0, t.min(), t.max()

    def __getitem__(self, item):
        key = self.wav[item].split('/')[-1]
        data, sample_rate = torchaudio.load(self.wav[item])
        label = torch.Tensor(self.df.loc[key].tolist()).unsqueeze(dim=0)
        data = torch.cat((data, torch.zeros(1, 384)), 1)
        mag_power = torchaudio.transforms.Spectrogram(n_fft=256, win_length=256, hop_length=128)(data)[:, :128, :128]
        mag_power_log = torchaudio.transforms.AmplitudeToDB(stype='power')(mag_power)
        mag_power_log, min, max = self.__normalize(mag_power_log)
        data = torch.stft(input=data, n_fft=256, win_length=256, hop_length=128, center=True)[:, :128, :128, :2]
        mag, phase = audioF.magphase(data)
        # if self.transform is not None:
        #     data = self.transform(data)

        return mag_power_log, phase, min, max, label
