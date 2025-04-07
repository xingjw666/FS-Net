import numpy as np
import torch
import tqdm

from dataset_u import *
from model import *


def cal_Dice(img1,img2):
    shape = img1.shape
    I = 0
    U = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] >= 1 and img2[i,j] >= 1:
                I += 1
            if img1[i,j] >= 1 or img2[i,j] >= 1:
                U += 1
    return 2*I/(I+U+1e-5)



def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    target = target.float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def dice_coef(y_true, y_pred):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (torch.sum(y_true_f) + torch.sum(y_pred_f))

def weighted_dice_with_CE(y_true, y_pred):
    CE = F.binary_cross_entropy(y_pred, y_true)
    return (1 - dice_coef(y_true, y_pred)) + CE

def cal_Dice(img1,img2):
    shape = img1.shape
    I = 0
    U = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] >= 1 and img2[i,j] >= 1:
                I += 1
            if img1[i,j] >= 1 or img2[i,j] >= 1:
                U += 1
    return 2*I/(I+U+1e-5)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
torch.cuda.empty_cache()

path = ''

all_data = Mydataset('', dataset='train')

data_loader = DataLoader(all_data, batch_size=1, shuffle=True)
net = unet_3().to(device)
opt = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5, betas=(0.9, 0.99))
small_size = 152
num_rows = 2
num_cols = 2
tnt1 = 0
out_image_save = np.zeros((30, 304, 304))
mask_image_save = np.zeros((30, 304, 304))
epoch = 0
all_loss = 1
opt = optim.Adam(net.parameters(), lr=0.00001, weight_decay=1e-5, betas=(0.9, 0.99))

while epoch <= 1000:

    all_loss = 0
    all_loss1 = 0
    all_dice1 = []
    all_j = []
    t = 0
    r = 0
    kwk = 0
    data_loader = tqdm(data_loader)
    for f, (origin_image, segment_image) in enumerate(data_loader):
        origin_image, segment_image = origin_image.float(), segment_image.float()
        origin_image = origin_image.permute(0, 1, 4, 3, 2)

        origin_image, segment_image = origin_image.to(device), segment_image.to(device)
        out_imge = torch.zeros(1, 1, 304, 304).to(device)
        mask1 = torch.zeros(1, 1, 304, 304).to(device)
        for i in range(num_rows):
            for j in range(num_cols):
                start_row = i * small_size
                end_row = (i + 1) * small_size
                start_col = j * small_size
                end_col = (j + 1) * small_size

                origin_image11 = origin_image[:, :, :, start_row:end_row, start_col:end_col]
                segment_image11 = segment_image[:, :, start_row:end_row, start_col:end_col]

                # Data enhancement
                xnn = np.random.randint(0, 4)
                origin_image11 = torch.rot90(origin_image11, k=xnn, dims=(3, 4))
                segment_image11 = torch.rot90(segment_image11, k=xnn, dims=(2, 3))
                xnnn = np.random.randint(0, 2)
                xnnnn = np.random.randint(0, 2)
                if xnnn == 0:
                    origin_image11 = torch.flip(origin_image11, dims=[4])
                    segment_image11 = torch.flip(segment_image11, dims=[3])
                if xnnnn == 0:
                    origin_image11 = torch.flip(origin_image11, dims=[3])
                    segment_image11 = torch.flip(segment_image11, dims=[2])


                out_img1111 = net(origin_image11)
                out_imge[:, :, start_row:end_row, start_col:end_col] = out_img1111
                mask1[:, :, start_row:end_row, start_col:end_col] = segment_image11

                train_loss = weighted_dice_with_CE(segment_image11, out_img1111)

                opt.zero_grad()
                train_loss.backward()
                opt.step()
                loss_aa = np.array(train_loss.item())
                all_loss = all_loss + loss_aa
                t += 1


        weight_path = os.path.join(path, format(str('WEIGHT') + '-' + str(epoch) + str('.pth')))

    net.eval()
    with torch.no_grad():
        out_img111 = torch.zeros(1, 1, 304, 304).to(device)
        path11 = ''

        all_test_data = Mydataset(path11, transform=None, dataset='test')

        data_loader_t = DataLoader(all_test_data, batch_size=1, shuffle=False)
        for y, (origin_image1, segment_image1) in enumerate(data_loader_t):
            origin_image1, segment_image1 = origin_image1.float(), segment_image1.float()
            origin_image1, segment_image1 = origin_image1.to(device), segment_image1.to(device)
            origin_image1 = origin_image1.permute(0, 1, 4, 3, 2)
            out_img111 = torch.zeros(1, 1, 304, 304).to(device)

            for i in range(num_rows):
                for j in range(num_cols):
                    start_row = i * small_size
                    end_row = (i + 1) * small_size
                    start_col = j * small_size
                    end_col = (j + 1) * small_size
                    origin_image11 = origin_image1[:, :, :, start_row:end_row, start_col:end_col]
                    out_img = net(origin_image11)
                    out_img111[:, :, start_row:end_row, start_col:end_col] = out_img

            pred = out_img111.cpu()
            pred = (np.squeeze(pred.clone().detach().numpy()))
            target = segment_image1.cpu()
            target = (np.squeeze(target.clone().detach().numpy()))
            train_loss_t = weighted_dice_with_CE(segment_image1, out_img111)

            out_image_save[y, :, :] = pred
            mask_image_save[y, :, :] = target
            dice_c = dice_coef(pred, target)
            all_dice1.append(dice_c)

            loss_aa_t = np.array(train_loss_t.item())
            all_loss1 = all_loss1 + loss_aa_t
            r += 1

    # return train
    net.train()

    m_loss = all_loss / t
    c_loss = all_loss1 / r
    all_dice1 = np.array(all_dice1)
    d_loss = np.mean(all_dice1)
    if tnt1 < d_loss:
        weight_path1 = os.path.join(path, format(str('best_model') + str('.pth')))
        torch.save(net.state_dict(), weight_path1)
        tnt1 = d_loss
    print(f'Epoch {epoch} ------ Average train loss: {m_loss}, Test loss: {c_loss}, Dice score: {d_loss}, Lowest score: {np.min(all_dice1)}')
    epoch += 1