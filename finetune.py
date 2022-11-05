#%%
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim import Adam, SGD
import torch.optim.lr_scheduler
import numpy as np

from score_function import KNN
from readfile import readfile
from model import new_resnet50, downstreamnet

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #%%

    backbone = new_resnet50()

    backbone.load_state_dict(torch.load('./save_sslmodel/best_model_state_dict.pt'))

    downstream = downstreamnet()

    finetune_model = torch.nn.Sequential(backbone, downstream).to(device)

    #%%
    epochs = 20

    optimizer = SGD(backbone.parameters(), 0.001, momentum=0.9)

    loss_fn = torch.nn.CrossEntropyLoss()

    #%%
    try:
        os.mkdir('./save_finetunemodel')
            
    except FileExistsError:
        pass

    #%%
    test_x, test_y = readfile('./data/test/', True)
    test_x, val_x, test_y, val_y = train_test_split(test_x, test_y, test_size=0.2)
    test_set = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))
    test_loader = DataLoader(test_set, batch_size=8, shuffle=True)

    #%%
    for epoch in range(epochs):

        finetune_model.train()
        true_y = []
        pred_y = []

        for x, y in test_loader:

            x = x.to(torch.float).to(device)
            y = y.to(torch.long).to(device)
            y_hat = finetune_model(x)

            loss = loss_fn(y_hat, y)    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            true_y.append(y.cpu())
            pred_y.append(y_hat.detach().argmax(dim=-1).cpu())

        true_y = torch.cat(true_y, dim=0)
        pred_y = torch.cat(pred_y, dim=0)
        train_acc = (true_y == pred_y).float().mean()

        ### eval ###
        finetune_model.eval()

        new_val_x = torch.tensor(val_x).to(torch.float).to(device)
        new_val_y = torch.tensor(val_y).to(torch.long).cpu()
        val_y_hat = finetune_model(new_val_x).detach().argmax(dim=-1).cpu()
        val_acc = (new_val_y == val_y_hat).float().mean()


        torch.save(backbone.state_dict(), ('./save_finetunemodel/'+ str(epoch) +'_model_state_dict.pt'))

        print(", ".join([f"Epoch {epoch:3d}/{epochs:3d}",f"train_acc: {train_acc:.4f}",f"val_acc: {val_acc:.4f}"]))

    #%%

    final_model = new_resnet50()

    final_model.load_state_dict(torch.load('./save_finetunemodel/0_model_state_dict.pt'))

    final_model = final_model.to(device)

    #%%

    try_set = TensorDataset(torch.tensor(val_x), torch.tensor(val_y))
    try_loader = DataLoader(try_set, batch_size=100, shuffle=False)

    #%%
    final_x = np.zeros((len(val_x), 512), dtype=np.float32)
    final_y = np.zeros((len(val_y)), dtype=np.float32)
    num = 0
    final_model.eval()
    for x, y in try_loader:

        try_y_hat = final_model(x.to(torch.float).to(device))
        final_x[num : num+100, :] = try_y_hat.cpu().detach().numpy()
        final_y[num : num+100] = y
        num +=100

    #%%
    acc = KNN(torch.tensor(final_x), torch.tensor(final_y), batch_size=16)
    print("Accuracy: %.5f" % acc)
# %%
