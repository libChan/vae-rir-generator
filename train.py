from RIRData import *
from torch import nn, optim
from torch.nn import functional as F
import torchaudio
import torchaudio.functional as audioF
from torchsummary import summary
import librosa
import librosa.display
from torch.utils.tensorboard.writer import SummaryWriter
from models.my_vae import MyVAE
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split

import warnings
import random
import argparse

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# Backend settings
torch.cuda.set_device(2)
torch.manual_seed(2)
device = torch.device("cuda")


def train(epoch, model, optimizer, writer, train_loader, args):
    model.train()
    train_loss = 0
    for batch_idx, (mag, phase, min, max, label) in enumerate(train_loader):
        mag = mag.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        kwargs = {'input':mag, 'label': label}
        [recon_batch, input, mu, logvar] = model(**kwargs)
        loss_dict = model.loss_function(recon_batch, input, mu, logvar)
        loss_dict['loss'].backward()
        train_loss += loss_dict['loss']
        optimizer.step()

        writer.add_scalar('loss', loss_dict['loss'], global_step=epoch * len(train_loader) + batch_idx)
        writer.add_scalar('recon_Loss', loss_dict['Reconstruction_Loss'],
                          global_step=epoch * len(train_loader) + batch_idx)
        writer.add_scalar('kl_Loss', loss_dict['KLD'], global_step=epoch * len(train_loader) + batch_idx)
        if batch_idx % args.log_interval == 0:
                rnd_generate(model=model, writer=writer, train_loader=train_loader, epoch=epoch, batch_idx=batch_idx, num=4, args=args)
            # spec_to_wav(epoch, batch_idx, mag, max=max, min=min)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss * args.batch_size / len(train_loader.dataset)))


def test(epoch, model, optimizer, writer, test_loader, train_loader, args):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (mag, phase, min, max, label) in enumerate(test_loader):
            mag = mag.to(device)
            label = label.to(device)
            phase = phase.to(device)
            kwargs = {'input': mag, 'label': label}
            [recon_batch, input, mu, logvar] = model(**kwargs)
            test_loss += model.loss_function(recon_batch, input, mu, logvar)['loss']

            if i == 0:
                n = 4
                phase = torch.exp(phase * torch.complex(real=torch.zeros(phase.shape[1], phase.shape[2]).to(device),
                                                        imag=torch.ones(phase.shape[1], phase.shape[2]).to(device)))
                origin_pad = F.pad(input=mag, pad=(0, 1, 0, 1), mode='replicate')
                recon_pad = F.pad(input=recon_batch, pad=(0, 1, 0, 1), mode='replicate')

                min_pad = torch.cat((min, torch.Tensor([-100.0])), dim=0)
                max_pad = torch.cat((max, torch.Tensor([10.0])), dim=0)

                # min_pad = F.pad(input=min, pad=(0,1), mode='replicate')
                # max_pad = F.pad(input=max, pad=(0, 1), mode='replicate')
                fig, ax = plt.subplots(nrows=2, ncols=n // 2, sharex=True)
                for i in range(n // 2):
                    rnd = random.randint(0, args.batch_size-1)
                    origin_m_inverse = inverse_normalize(origin_pad[rnd], min_pad[rnd], max_pad[rnd])
                    recon_m_inverse = inverse_normalize(recon_pad[rnd], min_pad[rnd], max_pad[rnd])
                    img1 = librosa.display.specshow(origin_m_inverse.squeeze(0).cpu().numpy(), y_axis='log',
                                                    sr=16000,
                                                    hop_length=128, x_axis='time', ax=ax[0, i])
                    img2 = librosa.display.specshow(recon_m_inverse.squeeze(0).cpu().numpy(), y_axis='log',
                                                    sr=16000,
                                                    hop_length=128, x_axis='time', ax=ax[1, i])
                fig.colorbar(img1, ax=ax[0, n // 2 - 1], format="%+2.f dB")
                fig.colorbar(img2, ax=ax[1, n // 2 - 1], format="%+2.f dB")
                writer.add_figure('compare', figure=fig, global_step=epoch * len(train_loader) + i)

    print('====> Test set loss: {:.4f}'.format(test_loss * args.batch_size / len(train_loader.dataset)))


def inverse_normalize(t, min, max):
    return ((t.cpu() + 1.0) * (max.cpu() - min.cpu())) / 2.0 + min.cpu()


def rnd_generate(model, writer, train_loader, epoch, batch_idx, num, args):
    sample = torch.randn(num, args.latent_dim).to(device)
    sample = model.decode(sample).cpu()
    min = -100.0 * torch.ones(num)
    max = torch.FloatTensor(num).normal_(mean=13.4350, std=6.9768)
    spec_to_wav(epoch, writer, train_loader, batch_idx, sample, max=max, min=min)


def spec_to_wav(epoch, writer, train_loader, batch_idx, sample, max, min):
    with torch.no_grad():
        data = F.pad(input=sample, pad=(0, 1, 0, 1), mode='replicate')  # (129,129)
        max_pad = max
        min_pad = min
        for i in range(list(max.shape)[0]):
            data_inverse = inverse_normalize(data[i].cpu(), min_pad[i], max_pad[i])
            data_inverse_power = audioF.DB_to_amplitude(data_inverse, power=1.0, ref=1.0)
            griffin = torchaudio.transforms.GriffinLim(n_fft=256, hop_length=128, win_length=256, power=2.0)
            griffin.train()
            wave = griffin(data_inverse_power)

            fig = plt.figure()
            plt.plot(wave.t().numpy())
            #writer.add_figure('./train_output/fig/sample_spec_'+str(i), figure=fig, global_step=epoch * len(train_loader) + batch_idx)
            #writer.add_audio('./train_output/audio/sample_wav_'+str(i), snd_tensor=wave, global_step=epoch * len(train_loader) + batch_idx)

    # torchaudio.save(filepath='RIR_result/sampleWAV_' + str(epoch) + '.wav', src=wave, sample_rate=16000)
def main():
    parser = argparse.ArgumentParser(description='spech trainp arguments')
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--log_path', type=str, default='./train_output/log/')
    parser.add_argument('--conditional', type=bool, default=True)
    parser.add_argument('--num_classes', type=int, default=4)
    args = parser.parse_args()

    writer = SummaryWriter(args.log_path)

    # load dataset
    dataset = BUTRIR(transform=None)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
            
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0) 
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = MyVAE(in_channels=1, latent_dim=args.latent_dim, num_classes=args.num_classes, conditional=args.conditional).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.Adadelta(model.parameters())
#    summary(model, input_size=(1, 128, 128))
    
    for epoch in range(1, args.epoch + 1):
        train(epoch=epoch, model=model, optimizer=optimizer, train_loader=train_loader,writer=writer,args=args)
        test(epoch=epoch, model=model, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader, writer=writer, args=args)


if __name__ == "__main__":
    main()
