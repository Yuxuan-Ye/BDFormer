import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
import cv2
from torch.nn.modules.loss import CrossEntropyLoss
import pandas as pd


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )



def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler



def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    plt.figure(figsize=(7,15))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()

def save_imgs_multitask(img, msk, msk_pred, contour_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    plt.figure(figsize=(7,15))

    plt.subplot(4,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(4,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(4,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    plt.subplot(4, 1, 4)
    plt.imshow(contour_pred, cmap='gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()

def save_msk_pred(msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)
    print("save path = " +  save_path + "outputs/" + str(i) + '.png')
    cv2.imwrite(save_path + "outputs/pred_masks/" + str(i) + '.png', msk_pred*255) # multi-task BDFormer
    # cv2.imwrite(save_path + "outputs/" + str(i) + '.png', msk_pred*255) # single-task and Xboundformer

def save_msk_contour(contour_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    cv2.imwrite(save_path + "outputs/pred_contours/" + str(i) + '.png', contour_pred*255)


class BCELoss(nn.Module):
    def __init__(self, weight=None):
        super(BCELoss, self).__init__()
        self.weight = weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, pred, target, weight=None):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.contiguous().view(size, -1)
        target_ = target.contiguous().view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


class DiceLoss_Contour(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_Contour, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


# single task tools
class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
       

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
        else: return image, mask
            

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
        else: return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
        else: return image, mask 


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'PH2':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
            
    def __call__(self, data):
        img, msk = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized, msk


# multi task tools; image - data input, mask - seg_label, contour - contour_label
class myToTensor_multi:  # why
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask, contour = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1), torch.tensor(contour).permute(2, 0, 1)


class myResize_multi:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask, contour = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w]), TF.resize(contour, [self.size_h, self.size_w])


class myRandomHorizontalFlip_multi:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask, contour = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask), TF.hflip(contour)
        else:
            return image, mask, contour


class myRandomVerticalFlip_multi:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask, contour = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask), TF.vflip(contour)
        else:
            return image, mask, contour


class myRandomRotation_multi:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p

    def __call__(self, data):
        image, mask, contour = data
        if random.random() < self.p:
            return TF.rotate(image, self.angle), TF.rotate(mask, self.angle), TF.rotate(contour, self.angle)
        else:
            return image, mask, contour


class myNormalize_multi:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'PH2':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748

    def __call__(self, data):
        img, msk, contour = data
        img_normalized = (img - self.mean) / self.std
        img_normalized = ((img_normalized - np.min(img_normalized))
                          / (np.max(img_normalized) - np.min(img_normalized))) * 255.
        return img_normalized, msk, contour


# caculate multi-task loss


def Caculate_multi_task_loss(pred_seg, pred_contour, mask_seg, mask_contour, alpha):
    ce_loss_contour = CrossEntropyLoss(weight=torch.tensor([1.0, 55.0]).cuda())
    # ce_loss_contour = CrossEntropyLoss()
    dice_loss_contour = DiceLoss_Contour(n_classes=2)
    bce_dice_loss = BceDiceLoss()
    # bce_loss = BCELoss()

    loss_seg = bce_dice_loss(pred_seg, mask_seg)

    pred_contour_soft = F.softmax(pred_contour, dim=1)
    loss_contour_ce = ce_loss_contour(pred_contour, mask_contour.long())
    loss_contour_dice = dice_loss_contour(pred_contour, mask_contour)
    loss_contour_mse = F.mse_loss(pred_contour_soft[:, 1, :, :], mask_contour.to(torch.float32))
    loss_contour = 0.4 * loss_contour_ce + 0.2 * loss_contour_dice + 0.4 * loss_contour_mse

    consistency_loss = alpha * Calculate_consistency_loss(pred_seg, pred_contour)

    loss = loss_seg + loss_contour + consistency_loss

    return loss


# consistency loss
def OnehotEncoder(input_tensor):
    tensor_list = []
    for i in range(2):
        temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def Calculate_consistency_loss(pred_seg, pred_contour):  # Batch_size >= 2
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss_Contour(n_classes=2)
    seg_img = torch.argmax(torch.softmax(pred_seg, dim=1), dim=1).squeeze(0)
    # seg_img = pred_seg
    contour_img = torch.argmax(torch.softmax(pred_contour, dim=1), dim=1).squeeze(0)
    loss_consistency = 0
    Batch_num = pred_seg.shape[0]

    for i in range(Batch_num):
        # calculate consistency between filled_contour and processed_seg
        # temp1 = pred_seg, temp2 = pred_contour_filled
        temp1 = seg_img[i].cpu().detach().numpy().astype('uint8')
        temp2 = contour_img[i].cpu().detach().numpy().astype('uint8')

        contours, hierarchy = cv2.findContours(temp1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(temp1, contours, -1, 1, cv2.FILLED)
        contours, hierarchy = cv2.findContours(temp2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        area = []
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        max_idx = np.argsort(np.array(area))  # 轮廓索引
        for idx in max_idx:
            cv2.drawContours(temp2, contours, idx, 1, cv2.FILLED)  # 按轮廓索引填充轮廓

        temp1 = torch.tensor(temp1, dtype=torch.float32)
        temp1 = temp1.reshape([1, 256, 256])
        temp1 = OnehotEncoder(temp1).cuda()
        temp2 = torch.tensor(temp2).cuda()
        temp2 = temp2.reshape([1, 256, 256])
        loss_consistency1_ce = ce_loss(temp1, temp2.long())
        loss_consistency1_dice = dice_loss(temp1, temp2[:], softmax=True)
        loss_consistency1 = 0.5*loss_consistency1_ce + 0.5*loss_consistency1_dice

        # calculate consistency between pred_contour and cv2.Canny(pred_seg)
        # temp3 = pred_seg process to contour. use kernel[5, 5] dilation
        temp3 = seg_img[i].cpu().detach().numpy().astype('uint8')
        temp3 = (cv2.Canny(temp3.astype(np.uint8), 1, 4) / 255).astype('uint8')
        # dilation
        kernel = np.ones((5, 5), np.uint8)
        temp3 = cv2.dilate(temp3, kernel)
        # end dilation
        temp3 = torch.tensor(temp3).cuda()
        temp3 = temp3.reshape([1, 256, 256])
        # temp4 = contour_img.cpu().detach().numpy().astype('uint8')
        temp4 = contour_img[i]
        temp4 = temp4.reshape([1, 256, 256])
        temp4 = OnehotEncoder(temp4)

        loss_contour_weight = torch.tensor([1.0, 55.0]).cuda()
        loss_consistency2_CE = F.cross_entropy(temp4, temp3[:].long(), weight=loss_contour_weight)
        output_contour_soft = F.softmax(temp4, dim=1)
        loss_consistency2_dice = dice_loss(temp4, temp3[:], softmax=True)
        loss_consistency2_mse = F.mse_loss(output_contour_soft[:, 1, :, :], temp3.to(torch.float32))
        loss_consistency2 = 0.4 * loss_consistency2_CE + 0.2 * loss_consistency2_dice + 0.4 * loss_consistency2_mse

        loss_consistency += loss_consistency1 + loss_consistency2
    loss_consistency = loss_consistency/Batch_num
    return loss_consistency



