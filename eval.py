#%%
import torch
from torch.utils.data import DataLoader
import numpy as np

from readfile import readfile
from model import new_resnet50

if __name__ == '__main__':
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #%%
    final_model = new_resnet50()

    final_model.load_state_dict(torch.load('./save_finetunemodel/2_model_state_dict.pt'))

    final_model = final_model.to(device)

    #%%
    train_x = readfile('./data/unlabeled/', False)
    batch_size = 128
    train_loader = DataLoader(train_x, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    final_x = np.zeros((len(train_x), 512), dtype=np.float32)
    num = 0

    #%%
    final_model.eval()
    for x in train_loader:

        try_y_hat = final_model(x.to(torch.float).to(device))
        final_x[num : num+batch_size, :] = try_y_hat.cpu().detach().numpy()
        num +=batch_size

    np.save('result', final_x)