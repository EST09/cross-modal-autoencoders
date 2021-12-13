import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from final_dataloader import ImageDataset
import final_model as AENet

import argparse
import numpy as np
import sys
import os
import imageio

# adapted from pytorch/examples/vae and ethanluoyc/pytorch-vae
 
# parse arguments
def setup_args():
 
    options = argparse.ArgumentParser()

    options.add_argument('--save-dir', action="store", dest="save_dir", default="my_save_dir")
    options.add_argument('-pt', action="store", dest="pretrained_file", default=None)
    options.add_argument('-bs', action="store", dest="batch_size", default = 128, type = int) 
    options.add_argument('-ds', action="store", dest="datadir", default = "data_folder/my_data/")
 
    options.add_argument('-iter', action="store", dest="max_iter", default = 800, type = int) #800
    options.add_argument('-lr', action="store", dest="lr", default=1e-3, type = float)
    options.add_argument('-nz', action="store", dest="nz", default=128, type = int)
    options.add_argument('-lamb', action="store", dest="lamb", default=0.0000001, type = float)
    options.add_argument('-lamb2', action="store", dest="lamb2", default=0.001, type = float)
    options.add_argument('--conditional', action="store_true")
 
    return options.parse_args()
 
args = setup_args()
os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(args.save_dir, "log.txt"), 'w') as f:
    print(args, file=f)

# retrieve dataloader
trainset = ImageDataset(datadir=args.datadir, mode='train')
testset = ImageDataset(datadir=args.datadir, mode='test')

train_loader = DataLoader(trainset, batch_size=args.batch_size, drop_last=False, shuffle=True)
test_loader = DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False)

print('Data loaded')

#VAE 

model = AENet.VAE(latent_variable_size=args.nz, batchnorm=True)





#cross_entropy
CE_weights = torch.FloatTensor([4.5, 0.5])
 


CE = nn.CrossEntropyLoss(CE_weights)
 

optimizer = optim.Adam([{'params': model.parameters()}], lr = args.lr)

def loss_function(recon_x, x, mu, logvar, latents):
    MSE = nn.MSELoss()
    lloss = MSE(recon_x,x)

    if args.lamb>0:
        KL_loss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        lloss = lloss + args.lamb*KL_loss

    return lloss
 
def train(epoch):
    
    model.train()

    train_loss = 0
    total_clf_loss = 0
    
    for batch_idx, samples in enumerate(train_loader):
 
        inputs = Variable(samples['image_tensor'])
        
        
        
 
        optimizer.zero_grad()
        recon_inputs, latents, mu, logvar = model(inputs)
        loss = loss_function(recon_inputs, inputs, mu, logvar, latents)
        train_loss += loss.data.item() * inputs.size(0)
        
        
        
        
        loss.backward()
        
        optimizer.step()
        

    with open(os.path.join(args.save_dir, "log.txt"), 'a') as f:
        print('Epoch: {} Average loss: {:.15f} Clf loss: {:.15f} '.format(epoch, train_loss / len(train_loader.dataset), total_clf_loss / len(train_loader.dataset)), file=f)
    

def test(epoch):
    model.eval()
    

    test_loss = 0
    total_clf_loss = 0

    for i, samples in enumerate(test_loader):
 
        inputs = Variable(samples['image_tensor'])
        
 
        recon_inputs, latents, mu, logvar = model(inputs)
        
        loss = loss_function(recon_inputs, inputs, mu, logvar, latents)
        test_loss += loss.data.item() * inputs.size(0)
        

    test_loss /= len(test_loader.dataset)
    total_clf_loss /= len(test_loader.dataset)

    with open(os.path.join(args.save_dir, "log.txt"), 'a') as f:
        print('Test set loss: {:.15f} Test clf loss: {:.15f}'.format(test_loss, total_clf_loss), file=f)
    
    return test_loss
 
 
def save(epoch):
    model_dir = os.path.join(args.save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(model_dir, str(epoch)+".pth"))
    
 
def generate_image(epoch):
    img_dir = os.path.join(args.save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    model.eval()

    for i in range(5):
        samples = train_loader.dataset[np.random.randint(30)]
        inputs = samples['image_tensor']
        inputs = Variable(inputs.view(1,1,64,64))
 
        
 
        recon_inputs, _, _, _ = model(inputs)
 
        imageio.imwrite(os.path.join(img_dir, "Train_epoch_%s_inputs_%s.jpg" % (epoch, i)), np.uint8(inputs.cpu().data.view(64,64).numpy()*255))
        imageio.imwrite(os.path.join(img_dir, "Train_epoch_%s_recon_%s.jpg" % (epoch, i)), np.uint8(recon_inputs.cpu().data.view(64,64).numpy()*255))
 
        samples = test_loader.dataset[np.random.randint(5)]
        inputs = samples['image_tensor']
        inputs = Variable(inputs.view(1,1,64,64))
 
        
 
        recon_inputs, _, _, _ = model(inputs)
 
        imageio.imwrite(os.path.join(img_dir, "Test_epoch_%s_inputs_%s.jpg" % (epoch, i)), np.uint8(inputs.cpu().data.view(64,64).numpy()*255))
        imageio.imwrite(os.path.join(img_dir, "Test_epoch_%s_recon_%s.jpg" % (epoch, i)), np.uint8(recon_inputs.cpu().data.view(64,64).numpy()*255))
 
# main training loop
generate_image(0)
save(0)
 
_ = test(0)

for epoch in range(args.max_iter):
    print(epoch)
    
    train(epoch)
    
    _ = test(epoch)
    
 
    if epoch % 10 == 1:
        generate_image(epoch)
        save(epoch)