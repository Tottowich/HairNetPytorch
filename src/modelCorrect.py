import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime
import os
from tqdm import tqdm
import sys
import argparse
from random_word import RandomWords
from torch.utils.data import DataLoader
# from HairNetPytorch.src.model import train
from dataloader import HairNetDataset
import random
rand_word_generator = RandomWords()
class HairNet(nn.Module):
    def __init__(self,weights=""):
        super(HairNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if weights != "":
            self.load_state_dict(torch.load(weights))
            print("Loaded pretrained model from: ", weights)

    def forward(self, x):
        # print(x.shape)
        x = self.encoder(x).squeeze()
        # print(x.shape)
        out = self.decoder(x)
        # print(f"pos shape: {pos.shape}")
        # print(f"curv shape: {curv.shape}")
        return out
class HairNetLoss(nn.Module):
    def __init__(self,alpha=1.0,beta=1.0,batch_size=1):
        self.alpha = alpha
        self.beta = beta
        self.bs = batch_size
        super(HairNetLoss, self).__init__()

    def forward(self, output, convdata, visweight):
        pos_loss = 0.0
        cur_loss = 0.0
        for i in range(0,32):
            for j in range(0,32):
                pos_loss += (visweight[:,:,i,j].reshape(1,-1).mm(torch.pow((convdata[:,:,0:3,i,j]-output[:,:,0:3,i,j]),2).reshape(-1, 3))).sum()
                cur_loss += (visweight[:,:,i,j].reshape(1,-1).mm(torch.pow((convdata[:,:,3,i,j]-output[:,:,3,i,j]),2).reshape(-1, 1))).sum()
        # print(pos_loss/1024.0, cur_loss/1024.0)       
        return (self.alpha*pos_loss + self.beta*cur_loss)/(1024.0*self.bs)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,activation='relu'):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() if activation == 'relu' else nn.Tanh()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()
        # Input shape is (3, 256, 256)
        self.conv1 = EncoderBlock(3, 32, 8, 2, 3)
        self.conv2 = EncoderBlock(32, 64, 8, 2, 3)
        self.conv3 = EncoderBlock(64, 128, 6, 2, 2)
        self.conv4 = EncoderBlock(128, 256, 4, 2, 1)
        self.conv5 = EncoderBlock(256, 256, 3, 1, 1)
        self.conv6 = EncoderBlock(256, 512, 4, 2, 1)
        self.conv7 = EncoderBlock(512, 512, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=8)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxpool(x)
        return x



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,activation=None,upsample=True):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride=stride, padding=padding)
        if activation is not None:
            self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
        else:
            self.activation = None
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        else:
            self.upsample = None
        
    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
class Decoder(nn.Module):
    def __init__(self,):
        super(Decoder, self).__init__()
        # Backbone linear layers:
        self.linear1 = nn.Linear(512, 1024) # (1,1024)
        self.linear2 = nn.Linear(1024, 4096) # (1,4096)
        # Reshape the linear layers to 256x4x4:
        # Backbone upsample layers:
        self.conv8 = DecoderBlock(256, 512, 3, 1, 1, "relu")
        self.conv9 = DecoderBlock(512, 512, 3, 1, 1, "relu")
        self.conv9 = DecoderBlock(512, 512, 3, 1, 1, "relu")
        # Decoder Strand Curvature:
        self.conv_curv1 = DecoderBlock(512, 512, 1, 1, 0, "relu",upsample=False)
        self.conv_curv2 = DecoderBlock(512, 512, 1, 1, 0, "tanh",upsample=False)
        self.conv_curv3 = DecoderBlock(512, 100, 1, 1, 0,upsample=False) # (300,32,32)
        self.final_mlp_curv = nn.Sequential(
            nn.Linear(100*32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 100*32*32*1),
            nn.ReLU(),
        )
        # Decoder Strand Position:
        self.conv_position1 = DecoderBlock(512, 512, kernel_size=1, stride=1, padding=0, activation="relu",upsample=False)
        self.conv_position2 = DecoderBlock(512, 512, 1, 1, 0, "tanh",upsample=False)
        self.conv_position3 = DecoderBlock(512, 300, 1, 1, 0, upsample=False)
        self.final_mlp_position = nn.Sequential(
            nn.Linear(300*32*32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 100*32*32*3),
            nn.ReLU(),
        )

    
    def forward(self, x):
        # Backbone Decoder:
        x = x.view(-1, 512)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = x.view(-1, 256, 4, 4)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv9(x)
        # Decoder Strand Curvature:
        curv_x = self.conv_curv1(x)
        curv_x = self.conv_curv2(curv_x)
        curv_x = self.conv_curv3(curv_x) # (1,100,32,32)
        curv_x = self.final_mlp_curv(curv_x.view(-1,100*32*32)).view(-1,100,1,32,32) # (1,100,1,32,32)
        # Decoder Strand Position:
        pos_x = self.conv_position1(x)
        pos_x = self.conv_position2(pos_x)
        pos_x = self.conv_position3(pos_x) # (1, 300, 32, 32)
        pos_x = self.final_mlp_position(pos_x.view(-1,300*32*32)).view(-1,100,3,32,32) # (1,100,3,32,32)
        return torch.cat((pos_x, curv_x), dim=2) # (1,300,4,32,32)
# class PositionLoss:
#     def __init__(self,):
#         super(PositionLoss, self).__init__()
#         self.mse = nn.MSELoss()
#     def __call__(self, pred, target):
#         return self.mse(pred, target)
# class CurvatureLoss:
#     def __init__(self,):
#         self.mse = nn.MSELoss()
#     def __call__(self, pred, target, mask):
#         return self.mse(pred*mask, target*mask)
# class TotalLoss:
#     def __init__(self,alpha=1.0,beta=1.0):
#         self.position_loss = PositionLoss()
#         self.curvature_loss = CurvatureLoss()
#         self.alpha = alpha
#         self.beta = beta
#     def __call__(self, pred, target):
#         pos_pred = pred[:,0:300,...]
#         curv_pred = pred[:,300:,...]
#         pos_target = target["position"]
#         curv_target = target["curvature"]

#         pos_loss = self.position_loss(pos_pred, pos_target)
#         curv_loss = self.curvature_loss(curv_pred, curv_target)
#         return pos_loss + curv_loss
class TotalLoss:
    def __init__(self,alpha=1.0,beta=1.0):
        self.alpha = alpha
        self.beta = beta
    def __call__(self, pos_pred,curv_pred, target,mask):
        # pos_pred = pred[:,0:300,...]
        # curv_pred = pred[:,300:,...]
        pos_target = target["position"]
        curv_target = target["curvature"]
        bs = pos_pred.size(0)
        pos_loss = self.position_loss(pos_pred, pos_target,mask)
        curv_loss = self.curvature_loss(curv_pred, curv_target,mask)
        return (self.alpha*pos_loss + self.beta*curv_loss)/(1024*bs)

    def curvature_loss(self,pred, target, mask):
        return torch.mean(mask*(pred - target)**2)
    def position_loss(self,pred, target,mask):
        return torch.mean(mask*(pred - target)**2)



def test_model():
    model = HairNet().to("cuda")
    loss = HairNetLoss()
    batch_size = 32
    x = torch.randn(batch_size,3,256,256).to("cuda")
    #output_curv = torch.randn(batch_size,100,1,32,32).to("cuda")
    output_pos = torch.randn(batch_size,100,5,32,32).to("cuda")
    vis_mask = torch.randn(batch_size,100,32,32).to("cuda")
    out = model(x)
    start = time.monotonic()
    out = model(x)
    end = time.monotonic()
    print(f"Inference time: {end-start:.5f}s")
    # assert pos.shape == output_pos.shape
    # assert curv.shape == output_curv.shape
    print(out.shape)
    #target_dict = {"position":output_pos,"curvature":output_curv}
    l = loss(out,output_pos,vis_mask)
    print(l)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="models/hairnet.pth")
    parser.add_argument("--mode", type=str, default="train",help="train, test or demo")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight for position loss")
    parser.add_argument("--beta", type=float, default=1.0,help="weight for curvature loss")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adam", help="adam or sgd")
    parser.add_argument("--lr", type=float, default=0.001, help="Starting learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="Learning rate decay")
    parser.add_argument("--lr_decay_interval", type=int, default=-1, help="Learning rate decay interval, -1 (default) for no decay")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--no_cuda", action="store_true", help="Do not use cuda")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--test_interval", type=int, default=5, help="Test interval (epochs)")
    parser.add_argument("--save_interval", type=int, default=5, help="Save interval")
    parser.add_argument("--save_dir", type=str, default="models", help="Save directory")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval to log to txt.")
    parser.add_argument("--weights", type=str, default="", help="Path to weights file")
    parser.add_argument("--name", type=str, default=f"{rand_word_generator.get_random_word()}_{rand_word_generator.get_random_word()}", help="Save directory")
    
    parser.add_argument("--data_dir", type=str, default="HairNet_training_data/data/", help="Data directory containing; convdata and data")
    # parser.add_argument("--image_path", type=str, default="images/image.png")
    # parser.add_argument("--output_path", type=str, default="images/output.png")
    
    opt = parser.parse_args()
    if opt.mode == "demo":
        assert opt.image_path is not None, "Image path is required for demo mode"
        assert opt.output_path is not None, "Output path is required for demo mode"
    if opt.seed != -1:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
    
    return opt
class Trainer:
    """
    Trainer encapsulates all the logic necessary for training the HairNet model.
    All hyperparameters are specified in the command line.
    """
    def __init__(self,opt) -> None:
        self.main_save_dir, self.weight_save_dir, self.plot_save_dir = self.generate_dirs(opt)
        print(f"Saving to {self.main_save_dir}")
        print(f"Saving weights to {self.weight_save_dir}")
        print(f"Saving plots to {self.plot_save_dir}")
        self.opt = opt
        if self.opt.weights != "":
            self.name = self.opt.weights.split("/")[-2]
        self.name = opt.name
        self.save_dir = opt.save_dir
        # self.data_dir = opt.data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu")
        self.dataset_train = HairNetDataset(f"../{opt.data_dir}", train_flag=True,noise_flag=1)
        self.dataloader = DataLoader(self.dataset_train, batch_size=opt.batch_size, shuffle=True,num_workers=opt.num_workers)
        self.dataset_test = HairNetDataset(f"../{opt.data_dir}", train_flag=False,noise_flag=0)
        self.dataloader_test = DataLoader(self.dataset_test, batch_size=opt.batch_size, shuffle=True,num_workers=opt.num_workers)
        
        self.model = HairNet(weights=opt.weights).to(self.device)
        #self.loss = TotalLoss(alpha=opt.alpha,beta=opt.beta)
        self.loss = HairNetLoss(alpha=opt.alpha,beta=opt.beta,batch_size=opt.batch_size)
        self.optimizer = self.get_optimizer(opt.optim)
        self.lr_scheduler = self.get_lr_scheduler()
        self.training_loss = []
        self.prev_average = 0
        self.prev_average_test = 0
        self.test_loss = []
        self.best_loss = np.inf
        self.log_file = f"{self.main_save_dir}/log.txt"
        #self.test_data()
    def test_data(self):
        start_loading = time.monotonic()
        data,target,vis_mask = next(iter(self.dataloader))
        end_loading = time.monotonic()
        print(f"Loading time: {end_loading-start_loading:.5f}s")
        print(f"Data shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Vis mask shape: {vis_mask.shape}")
        # print(f"Target[0] shape: {target[0]}")
        # print(f"Vismap: {vis_mask[0]}")
    def write_init_log_message(self):
        with open(self.log_file,"w") as f:
            for arg in vars(self.opt):
                f.write(f"{arg}:{getattr(self.opt,arg)}\n")
            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            f.write(f"Total parameters: {params}\n")
                
    def write_new_epoch(self,epoch):
        with open(self.log_file,"a") as f:
            now = datetime.now()
            f.write(f"-----------New Epoch: {epoch} @ {now.strftime('%H:%M:%S')}-----------\n")

    def write_log_message(self, message):
        with open(self.log_file,"a") as f:
            f.write(f"{message}\n")
    def plot_loss_continously(self,train=True):
        if train:
            #self.training_loss.remove(max(self.training_loss))
            #self.training_loss.remove(min(self.training_loss))
            plt.plot(self.training_loss, label="Training loss")
            plt.savefig(f"{self.plot_save_dir}/loss.png")
            plt.close()
        else:
            plt.plot(self.test_loss, label="Test loss")
            plt.savefig(f"{self.plot_save_dir}/test_loss.png")
            plt.close()
    # def close_log_file(self):
    #     self.log_file.close()
    def generate_dirs(self,opt):
        if not os.path.exists(f"{opt.save_dir}/{opt.name}"):
            os.makedirs(f"{opt.save_dir}/{opt.name}")
            main_save_dir = f"{opt.save_dir}/{opt.name}"
            os.makedirs(f"{main_save_dir}/weights")
            weight_save_dir = f"{main_save_dir}/weights"
            os.makedirs(f"{main_save_dir}/plots")
            plot_save_dir = f"{main_save_dir}/plots"
        return main_save_dir,weight_save_dir,plot_save_dir
    def get_optimizer(self,optim):
        if optim == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.opt.lr, momentum=self.opt.momentum, weight_decay=self.opt.weight_decay)
        elif optim == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer {optim}")
    def get_lr_scheduler(self):
        if self.opt.lr_decay_interval == -1:
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        else:
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.lr_decay_interval, gamma=self.opt.lr_decay)
    def train_step(self,epoch):
        self.model.train()
        # self.write_init_log_message()
        for batch_idx, (data, target, vismap) in tqdm(enumerate(self.dataloader),desc="Training",total=len(self.dataloader)):
            data, target,vismap = data.to(self.device), target.to(self.device), vismap.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.loss(out, target,vismap)
            loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step()
            self.training_loss.append(loss.item())
            if batch_idx % self.opt.log_interval == 0:
                self.prev_average = np.array(self.training_loss[-self.opt.log_interval:]).mean()
                self.write_log_message(f"Train Epoch: {epoch} [{batch_idx}/{len(self.dataloader)} ({100. * batch_idx / len(self.dataloader):.0f}%)]\tLoss: {self.prev_average:.5e}")
                self.plot_loss_continously(train=True)
    def save_model(self,epoch,best=False):
        if best:
            torch.save(self.model.state_dict(),f"{self.weight_save_dir}/best.pth")
            self.write_log_message(f"Saved model to {self.weight_save_dir}/best.pth @ Epoch {epoch}")
        else:
            self.write_log_message(f"Saved model to {self.weight_save_dir}/epoch_{epoch}.pth")
            torch.save(self.model.state_dict(),f"{self.weight_save_dir}/epoch_{epoch}.pth")
    def test_model(self,epoch):
        print(f"Testing model @ epoch {epoch}")
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target, vismap) in tqdm(enumerate(self.dataloader_test),desc=f"Testing Epoch: {epoch}",total=len(self.dataloader_test)):
                data, target, vismap = data.to(self.device), target.to(self.device), vismap.to(self.device)
                out = self.model(data)
                test_loss += self.loss(out, target,vismap).item()
        test_loss /= len(self.dataloader)
        better = test_loss < self.best_loss
        self.test_loss.append(test_loss)
        self.write_log_message(f"Test set epoch {epoch}: Average test loss: {test_loss:.5e}, Difference from last test: {test_loss-self.prev_average_test:.5e}, Difference from last train: {test_loss-self.prev_average:.5e}, Learning rate: {self.optimizer.param_groups[0]['lr']:.5e}")
        self.prev_average_test = test_loss
        self.plot_loss_continously(train=False)
        return better

    def train_model(self):
        self.write_init_log_message()
        better = False
        for epoch in tqdm(range(1, self.opt.epochs + 1), desc="Epochs", total=self.opt.epochs):
            self.write_new_epoch(epoch)
            self.train_step(epoch)
            if self.opt.test_interval != -1 and epoch % self.opt.test_interval == 0:
                better = self.test_model(epoch)
            if (self.opt.save_interval != -1 and epoch % self.opt.save_interval == 0) or better:
                self.save_model(epoch,better)
            self.lr_scheduler.step()
        now = datetime.now()
        self.write_log_message(f"-----------Finished Training @ {now.strftime('%H:%M:%S')}-----------\n")
    

    
        
    

if __name__ == "__main__":
    opt = parse_args()
    # test_model()
    trainer = Trainer(opt)
    trainer.train_model()