import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataloader import HairNetDataset
from modelCorrect import HairNet
from dataloader import load_root
from mpl_toolkits.mplot3d import Axes3D

def get_examples(data_dir:str,train=False,index=1):
    """
    Get Convdata, img and txt from data_dir/convdata, data_dir/png, data_dir/txt
    """
    dataset = HairNetDataset(data_dir, train_flag=train, noise_flag=0)
    current_img, current_convdata, current_visweight = dataset[index] # Convdata: (1,100,3,32,32) img: (1,3,256,256)
    return current_img, current_convdata, current_visweight


def show_convdata(convdata,root,mask,ax):
    """
    Convdata represents Hair: Visualize Strands 3D
    """
    # fig = plt.figure(figsize=(18, 6))
    # ax = plt.axes(projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1, 1)

    print(f"Mask shape: {mask.shape}")
    x_std = convdata[:,0,:,:].std()
    y_std = convdata[:,1,:,:].std()
    z_std = convdata[:,2,:,:].std()
    x_mean = convdata[:,0,:,:].mean()
    y_mean = convdata[:,1,:,:].mean()
    z_mean = convdata[:,2,:,:].mean()
    # Un normalize
    # convdata[:,0,:,:] = convdata[:,0,:,:]*x_std-x_mean
    # convdata[:,1,:,:] = convdata[:,1,:,:]*y_std-y_mean
    # convdata[:,2,:,:] = convdata[:,2,:,:]*z_std-z_mean
    # Center:
    convdata[:,0,:,:] = (convdata[:,0,:,:]-x_mean)
    convdata[:,1,:,:] = (convdata[:,1,:,:]-y_mean)
    convdata[:,2,:,:] = (convdata[:,2,:,:]-z_mean)

    print(f"X std: {x_std}")
    print(f"Y std: {y_std}")
    print(f"Z std: {z_std}")

    #for s in range(100):
    for i in range(32):
        for j in range(32):
            if mask[i,j]:
                #ax.scatter3D(convdata[:,0,i,j]+root[0,0], convdata[:,1,i,j]+root[0,1], convdata[:,2,i,j]+root[0,2], c='black', marker='o')
                ax.plot3D(convdata[:,0,i,j], convdata[:,1,i,j], convdata[:,2,i,j], c='lightblue')
import os
def create_obj(convdata,root,mask,index,train=False,predict=False):
    """
    Create obj file from convdata, root and mask
    """
    path = f"./obj/{'train' if train else 'test'}"
    if not os.path.exists(path):
        os.makedirs(path)
    obj_file = open(f"{path}/{'label' if not predict else 'pred'}_{index}.obj","w")
    obj_file.write("# HairNetPytorch\n")
    obj_file.write("# Created by HairNetPytorch\n")
    for s in range(100):
        for i in range(32):
            for j in range(32):
                if mask[i,j]:
                    obj_file.write("v "+str(convdata[s,0,i,j]+root[i,j,0])+" "+str(convdata[s,1,i,j]+root[i,j,1])+" "+str(convdata[s,2,i,j]+root[i,j,2])+"\n")
                    # obj_file.write("v "+str(convdata[s,:,i,j])+"\n")#+" "+str(convdata[s,1,i,j])+" "+str(convdata[s,2,i,j])+"\n")


def show_img(img,ax):
    """
    Show image
    """
    ax.imshow(cv2.cvtColor((img*255).numpy().astype(np.uint8).transpose(1,2,0), cv2.COLOR_BGR2RGB))
    ax.set_title('Image')

def visualize_from_set(data_dir:str,index=1,train=False):
    """
    Visualize Convdata, img and txt from data_dir/convdata, data_dir/png, data_dir/txt
    """
    
    img, convdata, visweight = get_examples(data_dir,train=train,index=index)
    root, mask = load_root(data_dir=data_dir)
    print(f"Image shape: {img.shape}")
    print(f"Convdata shape: {convdata.shape}")
    print(f"Visweight shape: {visweight.shape}")
    print(f"Root shape: {root.shape}")
    print(f"Mask shape: {mask.shape}")
    fig = plt.figure(figsize=(18, 6))
    fig.set_tight_layout(False)
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.05)
    plt.axis('off')
    show_img(img,plt.subplot(gs[0]))
    show_convdata(convdata,root,mask,plt.subplot(gs[1],projection='3d'))
    plt.show()
    create_obj(convdata,root,mask,index,train=train)
    # print("root:",root)
    # print("mask:",mask)
    # plt.imshow(cv2.cvtColor((img*255).numpy().astype(np.uint8).transpose(1,2,0), cv2.COLOR_BGR2RGB))
    # plt.show()
    # show_convdata(convdata,root,mask)
@torch.no_grad()
def visualize_model_pred(data_dir,train=False,index=3,weights=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HairNet(weights=weights).to(device=device) 
    img, convdata, visweight = get_examples(data_dir,train=train,index=index)
    root, mask = load_root(data_dir=data_dir)
    print(f"Image shape: {img.shape}")
    print(f"Convdata shape: {convdata.shape}")
    print(f"Visweight shape: {visweight.shape}")
    print(f"Root shape: {root.shape}")
    print(f"Mask shape: {mask.shape}")
    fig = plt.figure(figsize=(18, 6))
    fig.set_tight_layout(False)
    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.05)
    plt.axis('off')
    show_img(img,plt.subplot(gs[0]))
    show_convdata(convdata,root,mask,plt.subplot(gs[1],projection='3d'))
    create_obj(convdata,root,mask,index,train=train,predict=False)
    out_put = model(img.unsqueeze(0).to(device=device))
    print("Output shape:", out_put.shape)
    show_convdata(out_put[0].cpu().numpy(),root,mask,plt.subplot(gs[2],projection='3d'))
    create_obj(out_put[0].cpu().numpy(),root,mask,index,train=train,predict=True)
    plt.show()
if __name__ == "__main__":
    data_dir = "../HairNet_training_data/data"
    model_path = "./models/overfitting/weights/best.pth"
    visualize_model_pred(data_dir,train=True,index=0,weights=model_path)


    
