import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


###################################
DEVICE = torch.device("cpu")
batch_size = 64
x_dim = 784
hidden_size = 10

lr = 5e-3
epoch_num = 50
LOG = "test_result"
###################################

#Downloading mnist data
def prepare_data():
    train_data = datasets.MNIST(root='./', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='./', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class VariationalAutoEncoder(nn.Module):
    def __init__(self, batch_size, hidden_size=16, input_size=784, VQ=False) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.VQ = VQ

        # Encoder
        self.encodernet = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.mu_encoder = nn.Sequential(
            nn.Linear(64, self.hidden_size),
            nn.ReLU()
        )
        self.var_encoder = nn.Sequential(
            nn.Linear(64, self.hidden_size),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_size),
            nn.Sigmoid()
        )
    
    def encoder(self, x:torch.Tensor):
        x = x.reshape(-1, 784)
        x = self.encodernet(x)
        mu = self.mu_encoder(x)
        log_var = self.var_encoder(x)
        return mu, log_var
    
    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor):
        # reparameterization trick
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)

        return eps.mul(std) + mu


    def forward(self, x:torch.Tensor):
        # Encoder
        mu, log_var = self.encoder(x)
        latent = self.reparameterize(mu, log_var)

        #Decoder
        out = self.decoder(latent)
        out = out.reshape(-1, 1, 28, 28)

        return mu, log_var, out, latent

class LossFunction(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reconstruction = nn.BCELoss(size_average=False)
    
    def forward(self, recon_x:torch.Tensor, x:torch.Tensor, 
                mu:torch.Tensor, logvar:torch.Tensor)->torch.Tensor:
        # Reconstruction term
        BCE = self.reconstruction(recon_x, x)
        
        # KLD : 0.5 * sum( sigma^2 + mu^2 - ln(sigma^2) - 1 )
        KLD = 0.5 * torch.sum(logvar.exp() + mu.pow(2) - logvar - 1)

        return BCE + KLD

def train_epoch(train_loader:DataLoader, epoch:int, optimizer:torch.optim.Adam, VAE:VariationalAutoEncoder, loss_function:LossFunction, log:SummaryWriter):
    VAE.train()
    tloader = tqdm(train_loader)
    tloader.set_description(f'Training... Epoch: {epoch:03d}')
    tloader.set_postfix_str(f'loss: {0.0000:0.4f}')
    loss_epoch = .0
    fin_idx = 0
    for idx, [data, label] in enumerate(tloader):
        optimizer.zero_grad()

        mu, logvar, out, _ = VAE(data)
        loss = loss_function(out, data, mu, logvar)

        loss.backward()
        optimizer.step()

        loss_epoch += loss.data
        fin_idx = idx + 1
        tloader.set_postfix_str(f'loss: {loss_epoch/fin_idx:0.4f}')
    loss_epoch /= fin_idx
    log.add_scalar('loss/Train', loss_epoch, epoch)
    log.flush()

def test_epoch(test_loader:DataLoader, epoch:int, VAE:VariationalAutoEncoder, loss_function:LossFunction, log:SummaryWriter, STATUS="Testing"):
    VAE.eval()
    vloader = tqdm(test_loader)
    vloader.set_description(f'{STATUS}... Epoch: {epoch:03d}')
    vloader.set_postfix_str(f'loss: {0.0000:0.4f}')
    loss = .0
    fin_idx = 0
    ret = None
    for idx, [data, labels] in enumerate(vloader):
        mu, logvar, out, latent = VAE(data)
        loss_tmp = loss_function(out, data, mu, logvar)

        loss += loss_tmp.data
        fin_idx = idx + 1
        vloader.set_postfix_str(f'loss: {loss/fin_idx:0.4f}')
    loss /= fin_idx
    if STATUS == "Testing":
        log.add_scalar('loss/Test', loss, epoch)
    else:
        log.add_scalar('loss/Validation', loss, epoch)
    log.flush()
    
    if STATUS == "Testing":
        plot_latent(latent=latent, )
        show_img(data, out, labels)

def get_tensorboard(log_name):
    logdir = os.path.join(os.getcwd(), "logs", log_name)
    return SummaryWriter(log_dir=logdir)

def show_img(data:torch.Tensor, outputs:torch.Tensor, labels:torch.Tensor, end:int=5):
    out_img = torch.squeeze(outputs.data).numpy()
    org_img = torch.squeeze(data.data).numpy()
    labels = torch.squeeze(labels.data).numpy()

    for idx in range(end):
        label_l = labels[idx]
        plt.imshow(org_img[idx], cmap='gray')
        plt.savefig(f'{label_l}_org.png')
        plt.imshow(out_img[idx], cmap='gray')
        plt.savefig(f'{label_l}_con.png')

def train(train_loader:DataLoader, test_loader:DataLoader, VAE:VariationalAutoEncoder, epochs:int, logdir:str):
    optimizer = torch.optim.Adam(VAE.parameters(), lr=lr)
    loss_function = LossFunction()
    log = get_tensorboard(logdir)

    for epoch in range(1, epochs+1):
        train_epoch(train_loader=train_loader, epoch=epoch, optimizer=optimizer, VAE=VAE, loss_function=loss_function, log=log)

        with torch.no_grad():
            test_epoch(test_loader=test_loader, epoch=epoch, VAE=VAE, loss_function=loss_function, log=log, STATUS="Validating")
    
    with torch.no_grad():
        outputs = test_epoch(test_loader=test_loader, epoch=epoch, VAE=VAE, loss_function=loss_function, log=log)
    
    return outputs

def plot_latent(latent, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = latent.numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

if __name__ == "__main__":
    train_dataloader, test_dataloader = prepare_data()
    VAE = VariationalAutoEncoder(batch_size=batch_size, hidden_size=hidden_size, input_size=784, VQ=False)
    train(train_loader=train_dataloader, test_loader=test_dataloader, VAE=VAE, epochs=epoch_num, logdir=LOG)

    #show_img(test_dataloader, outputs, end=5)
