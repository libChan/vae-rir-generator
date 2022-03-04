from models.my_vae import MyVAE
import torchaudio
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from spec_train import inverse_normalize
import torchaudio.functional as audioF

# Backend settings
torch.cuda.set_device(1)
torch.manual_seed(2)
device = torch.device("cuda")
# para
latent_dim = 32
gen_num = 6000 

model = MyVAE(in_channels=1, latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load('./savedModel/vae.pth'))


def batch_generate(batch_size, model):
    with torch.no_grad():
        sample = torch.randn(batch_size, latent_dim).to(device)
        sample = model.decode(sample).cpu()
        min_pad = -100.0 * torch.ones(batch_size)
        max_pad = torch.FloatTensor(batch_size).normal_(mean=13.4350, std=6.9768)

        data = F.pad(input=sample, pad=(0, 1, 0, 1), mode='replicate')  # (129,129)

        for i in range(batch_size):
            data_inverse = inverse_normalize(data[i].cpu(), min_pad[i], max_pad[i])
            data_inverse_power = audioF.DB_to_amplitude(data_inverse, power=1.0, ref=1.0)
            griffin = torchaudio.transforms.GriffinLim(n_fft=256, hop_length=128, win_length=256, power=2.0)
            griffin.train()
            wave = griffin(data_inverse_power)

#            plt.figure()
#            plt.plot(wave.t().numpy())
#            plt.savefig('./train_output/fig/vae_rir_' + str(i) + '.jpg')
            torchaudio.save(filepath='./train_output/audio/vae_rir_' + str(i) + '.wav', src=wave, sample_rate=16000)
            print(str(i) + 'gen done')


if __name__ == "__main__":
    batch_generate(gen_num, model)
