import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torchsummary

###################################
DEVICE = torch.device("cpu")
batch_size = 64
x_dim = 784
in_channel = 1
hidden_dim = 256
hidden_size = 8

lr = 1e-3
epoch_num = 50
LOG = "cnn_vae"+"_"+str(epoch_num)
###################################

#Downloading mnist data
def prepare_data():
    train_data = datasets.MNIST(root='./', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='./', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=12, prefetch_factor=2**5)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=12)

    return train_loader, test_loader

class Flatten(nn.Module):
    def forward(self, input:torch.Tensor):
        return input.reshape(input.shape[0], -1)
    
class UnFlatten(nn.Module):
    def forward(self, input:torch.Tensor, size=128):
        return input.reshape(input.shape[0], size, 1, 1)
    
class Printing(nn.Module):
    def forward(self, x:torch.Tensor):
        print(x.shape)
        return x 

class VariationalAutoEncoder(nn.Module):
    def __init__(self, batch_size, hidden_size=8, hidden_dim=128, input_size=784, VQ=False) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.VQ = VQ

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2), #28x28 -> 13x13
            nn.ReLU(),
            #Printing(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), #13x13 -> 5x5
            nn.ReLU(),
            #Printing(),
            nn.Conv2d(64, self.hidden_dim, kernel_size=5), #5x5 -> 1x1
            nn.ReLU(),
            #Printing(),
            Flatten()
        )
        
        self.mu = nn.Linear(self.hidden_dim, self.hidden_size)
        self.var = nn.Linear(self.hidden_dim, self.hidden_size)

        # Decoder
        self.decode = nn.Linear(self.hidden_size, self.hidden_dim)
        self.decoder = nn.Sequential(
            UnFlatten(),
            #Printing(),
            nn.ConvTranspose2d(self.hidden_dim, 64, kernel_size=5), #1x1 -> 5x5
            nn.ReLU(),
            #Printing(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2), #5x5 -> 13x13
            nn.ReLU(),
            #Printing(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2), #13x13 -> 28x28
            #Printing(),
            nn.Sigmoid()
        )

        self.N = torch.distributions.Normal(0, 1)

    
    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor):
        # reparameterization trick
        std = torch.exp(logvar/2)

        return std*self.N.sample(mu.shape) + mu
    
    def bottleneck(self, h:torch.Tensor):
        mu, logvar = self.mu(h), self.var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encoding(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar
    
    def decoding(self, z)->torch.Tensor:
        z = self.decode(z)
        h = self.decoder(z)
        return h

    def forward(self, x:torch.Tensor):
        # Encoder
        latent, mu, log_var = self.encoding(x)

        #Decoder
        out = self.decoding(latent)

        out = out.reshape(-1, 1, 28, 28)

        return mu, log_var, out

class LossFunction(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def clip(self, data):
        eps = 1e-7
        return torch.clip(data, eps, 1 - eps)
    
    def forward(self, recon_x:torch.Tensor, x:torch.Tensor, 
                mu:torch.Tensor, logvar:torch.Tensor)->torch.Tensor:
        # Reconstruction term
        BCE = F.binary_cross_entropy(self.clip(recon_x), self.clip(x), reduction="sum")
        
        # KLD : 0.5 * sum( sigma^2 + mu^2 - ln(sigma^2) - 1 )
        KLD = 0.5 * torch.sum(torch.exp(logvar/2) + mu.pow(2) - logvar - 1)

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

        mu, logvar, out = VAE(data)
        loss = loss_function(out, data, mu, logvar)
        loss /= batch_size

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
        mu, logvar, out = VAE(data)
        loss_tmp = loss_function(out, data, mu, logvar)
        loss_tmp /= batch_size
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
        show_img(data, out, labels, 10)

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
        plt.savefig(f'CNN_VAE_img/{label_l}_org.png')
        plt.imshow(out_img[idx], cmap='gray')
        plt.savefig(f'CNN_VAE_img/{label_l}_con.png')

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


if __name__ == "__main__":
    torch.manual_seed(2024)
    train_dataloader, test_dataloader = prepare_data()
    VAE = VariationalAutoEncoder(batch_size=batch_size, hidden_size=hidden_size, input_size=784, VQ=False)
    torchsummary.summary(VAE, (1, 28, 28))
    train(train_loader=train_dataloader, test_loader=test_dataloader, VAE=VAE, epochs=epoch_num, logdir=LOG)

    #show_img(test_dataloader, outputs, end=5)
x_dim = 784
in_channel = 1
hidden_dim = 256
hidden_size = 8

lr = 1e-3
epoch_num = 50
LOG = "cnn_vae"+"_"+str(epoch_num)
###################################

#Downloading mnist data
def prepare_data():
    train_data = datasets.MNIST(root='./', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='./', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=12)

    return train_loader, test_loader

class Flatten(nn.Module):
    def forward(self, input:torch.Tensor):
        return input.reshape(input.shape[0], -1)
    
class UnFlatten(nn.Module):
    def forward(self, input:torch.Tensor, size=128):
        return input.reshape(input.shape[0], size, 1, 1)
    
class Printing(nn.Module):
    def forward(self, x:torch.Tensor):
        print(x.shape)
        return x 

class VariationalAutoEncoder(nn.Module):
    def __init__(self, batch_size, hidden_size=8, hidden_dim=128, input_size=784, VQ=False) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.VQ = VQ

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2), #28x28 -> 13x13
            nn.ReLU(),
            #Printing(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2), #13x13 -> 5x5
            nn.ReLU(),
            #Printing(),
            nn.Conv2d(64, self.hidden_dim, kernel_size=5), #5x5 -> 1x1
            nn.ReLU(),
            #Printing(),
            Flatten()
        )
        
        self.mu = nn.Linear(self.hidden_dim, self.hidden_size)
        self.var = nn.Linear(self.hidden_dim, self.hidden_size)

        # Decoder
        self.decode = nn.Linear(self.hidden_size, self.hidden_dim)
        self.decoder = nn.Sequential(
            UnFlatten(),
            #Printing(),
            nn.ConvTranspose2d(self.hidden_dim, 64, kernel_size=5), #1x1 -> 5x5
            nn.ReLU(),
            #Printing(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2), #5x5 -> 13x13
            nn.ReLU(),
            #Printing(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2), #13x13 -> 28x28
            #Printing(),
            nn.Sigmoid()
        )

        self.N = torch.distributions.Normal(0, 1)

    
    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor):
        # reparameterization trick
        std = torch.exp(logvar/2)

        return std*self.N.sample(mu.shape) + mu
    
    def bottleneck(self, h:torch.Tensor):
        mu, logvar = self.mu(h), self.var(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encoding(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar
    
    def decoding(self, z)->torch.Tensor:
        z = self.decode(z)
        h = self.decoder(z)
        return h

    def forward(self, x:torch.Tensor):
        # Encoder
        latent, mu, log_var = self.encoding(x)

        #Decoder
        out = self.decoding(latent)

        out = out.reshape(-1, 1, 28, 28)

        return mu, log_var, out

class LossFunction(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def clip(self, data):
        eps = 1e-7
        return torch.clip(data, eps, 1 - eps)
    
    def forward(self, recon_x:torch.Tensor, x:torch.Tensor, 
                mu:torch.Tensor, logvar:torch.Tensor)->torch.Tensor:
        # Reconstruction term
        BCE = F.binary_cross_entropy(self.clip(recon_x), self.clip(x), reduction="sum")
        
        # KLD : 0.5 * sum( sigma^2 + mu^2 - ln(sigma^2) - 1 )
        KLD = 0.5 * torch.sum(torch.exp(logvar/2) + mu.pow(2) - logvar - 1)

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
        data = data.to(DEVICE)
        mu, logvar, out = VAE(data)
        loss = loss_function(out, data, mu, logvar)
        loss /= batch_size

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
    for idx, [data, labels] in enumerate(vloader):
        data = data.to(DEVICE)
        mu, logvar, out = VAE(data)
        loss_tmp = loss_function(out, data, mu, logvar)
        loss_tmp /= batch_size
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
        show_img(data, out, labels, 10)

def get_tensorboard(log_name):
    logdir = os.path.join(os.getcwd(), "logs", log_name)
    return SummaryWriter(log_dir=logdir)

def show_img(data:torch.Tensor, outputs:torch.Tensor, labels:torch.Tensor, end:int=5):
    data = data.cpu().detach()
    out_img = torch.squeeze(outputs.data).numpy()
    org_img = torch.squeeze(data.data).numpy()
    labels = torch.squeeze(labels.data).numpy()

    for idx in range(end):
        label_l = labels[idx]
        plt.imshow(org_img[idx], cmap='gray')
        plt.savefig(f'CNN_VAE_img/{label_l}_org.png')
        plt.imshow(out_img[idx], cmap='gray')
        plt.savefig(f'CNN_VAE_img/{label_l}_con.png')

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


if __name__ == "__main__":
    torch.manual_seed(2024)
    train_dataloader, test_dataloader = prepare_data()
    VAE = VariationalAutoEncoder(batch_size=batch_size, hidden_size=hidden_size, input_size=784, VQ=False).to(device=DEVICE)
    torchsummary.summary(VAE, (1, 28, 28))
    train(train_loader=train_dataloader, test_loader=test_dataloader, VAE=VAE, epochs=epoch_num, logdir=LOG)

    #show_img(test_dataloader, outputs, end=5)