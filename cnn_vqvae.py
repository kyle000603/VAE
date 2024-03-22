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
os.environ["CUDA_VISIBLE_DEVICES"]="0"
DEVICE = torch.device("cuda")
batch_size = 128
in_channel = 3
hidden_dim = 256
hidden_size = 10
lr = 2e-5
epoch_num = 200
BETA = 0.25

LOG = "cnn_vqvae"+"_"+str(epoch_num)
###################################

#Downloading mnist data
def prepare_data():
    transform_c = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])])
    train_data = datasets.CIFAR10(root='./', train=True, download=True, transform=transform_c)
    test_data = datasets.CIFAR10(root='./', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=12, prefetch_factor=2**5)
    test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, num_workers=12, prefetch_factor=2**5)

    return train_loader, test_loader
    
class Printing(nn.Module):
    def forward(self, x:torch.Tensor):
        print(x.shape)
        return x 
    
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1) -> None:
        super(ResidualBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.relu = nn.ReLU(True)

        self.weight_init()

    def weight_init(self):
        for module in self._modules:
            if isinstance(self._modules[module], nn.Conv2d):
                nn.init.kaiming_normal(self._modules[module].weight)
                try: self._modules[module].bias.data.fill_(0)
                except:pass
    
    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.conv3x3(out)
        out = self.relu(out)
        out = self.conv1x1(out)
        out += residual

        return out
    
class Embedding(nn.Module):
    def __init__(self, k_dim, e_dim, beta=0.25) -> None:
        super(Embedding, self).__init__()
        self.k_dim = k_dim
        self.e_dim = e_dim
        self.beta = beta
        self.loss_func = nn.MSELoss()

        self.embedding = nn.Embedding(k_dim, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / k_dim, 1.0 / k_dim)

    def forward(self, z:torch.Tensor):
        """
        z = [B, C, H, W]
        """
        z = z.permute(0, 2, 3, 1).contiguous() # [B, C, H, W] -> [B, H, W, C]
        z_flatten = z.view(-1, self.e_dim)

        # distance : ||z - e||2 = z^2 + e^2 - 2 e*z
        distance = torch.sum(z_flatten **2, dim=1, keepdim=True) \
            + torch.sum(self.embedding.weight**2, dim=1) \
                - 2*torch.matmul(z_flatten, self.embedding.weight.T)
        # argmin
        min_enc_idx = torch.argmin(distance, dim=1).unsqueeze(1)
        min_enc = torch.zeros(min_enc_idx.shape[0], self.k_dim).to(DEVICE)
        min_enc.scatter_(1, min_enc_idx, 1)

        # Quantized Latent Vector (Discrete)
        vq_z = torch.matmul(min_enc, self.embedding.weight).view(z.shape)

        # Loss (Embedding Loss + BETA * Commitment Loss)
        loss = self.loss_func(vq_z.detach(), z) + self.beta * self.loss_func(vq_z, z.detach())
        vq_z = z + (vq_z - z).detach()
        vq_z = vq_z.permute(0, 3, 1, 2)
        return loss, vq_z

class VQVariationalAutoEncoder(nn.Module):
    def __init__(self, k_dim=256, e_dim=8*8, h_dim=256) -> None:
        super(VQVariationalAutoEncoder, self).__init__()
        self.k_dim = k_dim
        self.e_dim = e_dim
        self.h_dim = h_dim

        self.encoder = nn.Sequential(
            #Printing(),
            nn.Conv2d(in_channels=3, out_channels=h_dim, kernel_size=4, stride=2, padding=1),       # 3x32x32 -> Hx16x16
            #Printing(),
            nn.Conv2d(in_channels=h_dim, out_channels=h_dim, kernel_size=4, stride=2, padding=1),   # Hx16x16 -> Hx8x8
            #Printing(),
            ResidualBlock(in_planes=h_dim, out_planes=h_dim),                                       # Hx8x8 -> Hx8x8
            #Printing(),
            ResidualBlock(in_planes=h_dim, out_planes=h_dim)                                       # Hx8x8 -> Hx8x8
            #Printing(),
            #,nn.Conv2d(in_channels=h_dim, out_channels=k_dim, kernel_size=1, stride=1)
            #Printing()
        )
        self.embedding = Embedding(k_dim=k_dim, e_dim=e_dim, beta=BETA)

        self.decoder = nn.Sequential(
            #Printing(),
            #nn.ConvTranspose2d(in_channels=k_dim, out_channels=h_dim, kernel_size=1, stride=1),
            #Printing(),
            ResidualBlock(in_planes=h_dim, out_planes=h_dim),
            #Printing(),
            ResidualBlock(in_planes=h_dim, out_planes=h_dim),
            #Printing(),
            nn.ConvTranspose2d(in_channels=h_dim, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
            #Printing(),
            nn.ConvTranspose2d(in_channels=h_dim, out_channels=3, kernel_size=4, stride=2, padding=1),
            #Printing(),
            nn.Tanh()
        )
        self.weight_init()
    
    def weight_init(self):
        for module in self._modules:
            if module == 'embedding':
                continue
            for m in self._modules[module]:
                if isinstance(m,nn.Conv2d):
                    nn.init.kaiming_normal(m.weight)
                    try:m.bias.data.fill_(0)
                    except:pass
        

    def forward(self, x):
        z = self.encoder(x)
        embedding_loss, vq_z = self.embedding(z)
        recon_x = self.decoder(vq_z)

        return recon_x, embedding_loss
        

class LossFunction(nn.Module):

    def __init__(self) -> None:
        super(LossFunction, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def clip(self, data):
        eps = 1e-7
        return torch.clip(data, eps, 1 - eps)
    
    def forward(self, recon_x:torch.Tensor, x:torch.Tensor)->torch.Tensor:
        # Reconstruction term
        return self.mse_loss(recon_x, x)

def train_epoch(train_loader:DataLoader, epoch:int, optimizer:torch.optim.Adam, model:VQVariationalAutoEncoder, loss_function:LossFunction, log:SummaryWriter):
    model.train()
    tloader = tqdm(train_loader)
    tloader.set_description(f'Training... Epoch: {epoch:03d}')
    tloader.set_postfix_str(f'loss: {.0:0.4f}, recon_loss: {.0:0.4f}, emb_loss: {.0:0.4f}')
    loss_epoch = .0
    fin_idx = 0
    recon_loss_epoch = .0
    embedding_loss_epoch = .0
    for idx, [data, label] in enumerate(tloader):
        data = data.to(DEVICE)
        optimizer.zero_grad()
    
        recon_x, embedding_loss = model(data)
        recon_loss = loss_function(recon_x, data)
        loss = recon_loss + embedding_loss
        #loss /= batch_size

        loss.backward()
        optimizer.step()

        recon_loss_epoch += recon_loss.data
        embedding_loss_epoch += embedding_loss.data
        loss_epoch += loss.data
        fin_idx = idx + 1
        tloader.set_postfix_str(f'loss: {loss_epoch/fin_idx:0.4f}, recon_loss: {recon_loss_epoch/fin_idx:0.4f}, emb_loss: {embedding_loss_epoch/fin_idx:0.4f}')
    loss_epoch /= fin_idx
    recon_loss_epoch /= fin_idx
    embedding_loss_epoch /= fin_idx
    log.add_scalar('loss/Train', loss_epoch, epoch)
    log.add_scalar('recon_loss/Train', recon_loss_epoch, epoch)
    log.add_scalar('embedding_loss/Train', embedding_loss_epoch, epoch)
    log.flush()

def test_epoch(test_loader:DataLoader, epoch:int, model:VQVariationalAutoEncoder, loss_function:LossFunction, log:SummaryWriter, STATUS="Testing"):
    model.eval()
    vloader = tqdm(test_loader)
    vloader.set_description(f'{STATUS}... Epoch: {epoch:03d}')
    vloader.set_postfix_str(f'loss: {.0:0.4f}, recon_loss: {.0:0.4f}, emb_loss: {.0:0.4f}')
    loss = .0
    fin_idx = 0
    recon_loss_epoch = .0
    embedding_loss_epoch = .0
    for idx, [data, labels] in enumerate(vloader):
        data = data.to(DEVICE)
        recon_x, embedding_loss = model(data)
        recon_loss = loss_function(recon_x, data)
        loss_tmp = recon_loss + embedding_loss
        #loss_tmp /= batch_size
        loss += loss_tmp.data
        recon_loss_epoch += recon_loss.data
        embedding_loss_epoch += embedding_loss.data
        fin_idx = idx + 1
        vloader.set_postfix_str(f'loss: {loss/fin_idx:0.4f}, recon_loss: {recon_loss_epoch/fin_idx:0.4f}, emb_loss: {embedding_loss_epoch/fin_idx:0.4f}')
    loss /= fin_idx
    recon_loss_epoch /= fin_idx
    embedding_loss_epoch /= fin_idx
    if STATUS == "Testing":
        log.add_scalar('loss/Test', loss, epoch)
        log.add_scalar('recon_loss/Test', recon_loss_epoch, epoch)
        log.add_scalar('embedding_loss/Test', embedding_loss_epoch, epoch)
    else:
        log.add_scalar('loss/Validation', loss, epoch)
        log.add_scalar('recon_loss/Validation', recon_loss_epoch, epoch)
        log.add_scalar('embedding_loss/Validation', embedding_loss_epoch, epoch)
    log.flush()
    
    if STATUS == "Testing":
        show_img(data, recon_x, labels)

def get_tensorboard(log_name):
    logdir = os.path.join(os.getcwd(), "logs", log_name)
    return SummaryWriter(log_dir=logdir)

def show_img(data:torch.Tensor, outputs:torch.Tensor, labels:torch.Tensor):
    out_img = outputs.permute(0,2,3,1).cpu().data.numpy()
    org_img = data.permute(0,2,3,1).cpu().data.numpy()
    labels = labels.data.numpy()

    for idx in range(org_img.shape[0]):
        label_l = labels[idx]
        plt.imshow(org_img[idx], cmap='gray')
        plt.savefig(f'CNN_VQVAE_img/{idx}_org{label_l}.png')
        plt.imshow(out_img[idx], cmap='gray')
        plt.savefig(f'CNN_VQVAE_img/{idx}_con{label_l}.png')

def train(train_loader:DataLoader, test_loader:DataLoader, model:VQVariationalAutoEncoder, epochs:int, logdir:str):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = LossFunction()
    log = get_tensorboard(logdir)

    for epoch in range(1, epochs+1):
        train_epoch(train_loader=train_loader, epoch=epoch, optimizer=optimizer, model=model, loss_function=loss_function, log=log)

        with torch.no_grad():
            test_epoch(test_loader=test_loader, epoch=epoch, model=model, loss_function=loss_function, log=log, STATUS="Validating")
    
    with torch.no_grad():
        outputs = test_epoch(test_loader=test_loader, epoch=epoch, model=model, loss_function=loss_function, log=log)
    
    return outputs


if __name__ == "__main__":
    torch.manual_seed(2024)
    train_dataloader, test_dataloader = prepare_data()
    model = VQVariationalAutoEncoder().to(DEVICE)
    #torchsummary.summary(model, (3, 32, 32))
    train(train_loader=train_dataloader, test_loader=test_dataloader, model=model, epochs=epoch_num, logdir=LOG)