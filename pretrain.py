#%%
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

from score_function import xt_xent
from readfile import readfile
from model import new_resnet50, projectionnet

#%%
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #%%
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.Pad(padding=(0, 0, 8, 16), fill=128),
        transforms.RandomResizedCrop((96, 96), scale=(0.6, 1.0)),
    ])

    #%%
    encoder_model = new_resnet50()

    projection_net = projectionnet()

    ssl_model = nn.Sequential(encoder_model, projection_net).to(device)
    #%%
    train_x = readfile('./data/unlabeled/', False)
    train_loader = DataLoader(train_x, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    epochs = 100
    optimizer = Adam(encoder_model.parameters(), 0.001)

    #%%
    try:
        os.mkdir('./save_sslmodel')
            
    except FileExistsError:
        pass

    #%%
    final_loss = 99999

    for epoch in range(0, epochs):
        ssl_model.train()
        
        for x in train_loader:

            y1 = transform(x).to(torch.float).to(device)
            y2 = transform(x).to(torch.float).to(device)

            y1 = ssl_model(y1)
            y2 = ssl_model(y2)

            loss = xt_xent(y1, y2)    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(encoder_model.state_dict(),  ('./save_sslmodel/'+ str(epoch) +'_model_state_dict.pt'))

        if final_loss > loss: 
            torch.save(encoder_model.state_dict(), './save_sslmodel/best_model_state_dict.pt')
            final_loss = loss

        print(", ".join([f"Epoch {epoch:3d}/{epochs:3d}",f"train_loss: {loss:.4f}",]))

