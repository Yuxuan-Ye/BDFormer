import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs, save_imgs_multitask, Caculate_multi_task_loss, save_msk_pred, save_msk_contour
import torch.nn as nn


# multi-task
def train_one_epoch_multi(train_loader,
                          model,
                          criterion,
                          optimizer,
                          scheduler,
                          epoch,
                          logger,
                          config,
                          scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    coefficient_generator = nn.Sigmoid()
    model.train()

    loss_list = []
    alpha = coefficient_generator(torch.tensor(epoch / 100 * 20 - 10))
    # alpha = 1
    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, msk_seg, msk_contour = data
        images, msk_seg, msk_contour = images.cuda(non_blocking=True).float(), msk_seg.cuda(non_blocking=True).float(), msk_contour.cuda(non_blocking=True).float()
        msk_contour = msk_contour.squeeze(dim=1)
        if config.amp:
            with autocast():
                pred_seg, pred_contour = model(images)
                loss = Caculate_multi_task_loss(pred_seg, pred_contour, msk_seg, msk_contour, alpha)
                # loss = criterion(out, msk_seg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_seg, pred_contour = model(images)
            loss = Caculate_multi_task_loss(pred_seg, pred_contour, msk_seg, msk_contour, alpha)
            # loss = criterion(out, msk_seg)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()


def val_one_epoch_multi(test_loader,
                        model,
                        criterion,
                        epoch,
                        logger,
                        config):
    # switch to evaluate mode
    # model = model
    model.eval()
    preds = []
    gts = []
    loss_list = []
    alpha = 1
    # alpha = 0
    # coefficient_generator = nn.Sigmoid()
    # alpha = coefficient_generator(torch.tensor(epoch / 100 * 20 - 15))
    # alpha = coefficient_generator(torch.tensor(epoch / 100 * 20 - 5))
    with torch.no_grad():
        for data in tqdm(test_loader, ncols=70):
            img, msk_seg, msk_contour = data
            img, msk_seg, msk_contour = img.cuda(non_blocking=True).float(), msk_seg.cuda(non_blocking=True).float(), msk_contour.cuda(non_blocking=True).float()
            pred_seg, pred_contour = model(img)
            msk_contour = msk_contour.squeeze(dim=1)
            # loss = criterion(pred_seg, msk_seg)
            loss = Caculate_multi_task_loss(pred_seg, pred_contour, msk_seg, msk_contour, alpha)
            loss_list.append(loss.item())
            gts.append(msk_seg.squeeze(1).cpu().detach().numpy())
            if type(pred_seg) is tuple:
                pred_seg = pred_seg[0]
            pred_seg = pred_seg.squeeze(1).cpu().detach().numpy()
            preds.append(pred_seg)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch_multi(test_loader,
                         model,
                         criterion,
                         logger,
                         config,
                         test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    # file = open("IouForImage.txt", "w")
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, ncols=70)):
            img, msk_seg, msk_contour = data
            img, msk_seg, msk_contour = img.cuda(non_blocking=True).float(), msk_seg.cuda(non_blocking=True).float(), msk_contour.cuda(non_blocking=True).float()
            pred_seg, pred_contour = model(img)
            loss = criterion(pred_seg, msk_seg)
            loss_list.append(loss.item())
            msk_seg = msk_seg.squeeze(1).cpu().detach().numpy()
            gts.append(msk_seg)
            if type(pred_seg) is tuple:
                pred_seg = pred_seg[0]
            pred_seg = pred_seg.squeeze(1).cpu().detach().numpy()
            pred_contour = torch.argmax(pred_contour, dim=1).squeeze(0).cpu().detach().numpy()
            preds.append(pred_seg)
            if i % config.save_interval == 0:
                # save_imgs(img, msk_seg, pred_seg, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                #           test_data_name=test_data_name)
                # # output contains mask and contour
                # save_imgs_multitask(img, msk_seg, pred_seg, pred_contour, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                #           test_data_name=test_data_name)
                ##### use when prediction #####
                save_imgs_multitask(img, msk_seg, pred_seg, pred_contour, i, config.work_dir,
                                    config.datasets, config.threshold,
                                    test_data_name=test_data_name)
                ##### use when save mask #####
                save_msk_pred(pred_seg, i, config.work_dir, config.datasets, config.threshold, test_data_name=test_data_name)
                save_msk_contour(pred_contour, i, config.work_dir, config.datasets, config.threshold, test_data_name=test_data_name)

        # file.close()
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)

# deep supervision for TransFuse
def train_one_epoch_deepsup(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        if config.amp:
            with autocast():
                out = model(images)

                loss = 0
                for i in range(len(out)):
                    loss += criterion(out[i], targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = 0
            for i in range(len(out)):
                loss += criterion(out[i], targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()


def val_one_epoch_deepsup(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader, ncols=70):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)

            loss = 0
            for i in range(len(out)):
                loss += criterion(out[i], msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch_deepsup(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, ncols=70)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)

            loss = 0
            for j in range(len(out)):
                loss += criterion(out[j], msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            if i % config.save_interval == 0:
                # save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                #           test_data_name=test_data_name)
                # #### use in predictions ####
                # save_imgs(img, msk, out, i, config.work_dir, config.datasets, config.threshold,
                #           test_data_name=test_data_name)
                #### use in save imgs ####
                save_msk_pred(out, i, config.work_dir, config.datasets, config.threshold, test_data_name=test_data_name)



        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


# xbound sturcture loss
def structure_loss(pred, mask):
    """            TransFuse train loss        """
    """            Without sigmoid             """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

# xboundformer
def train_one_epoch_xbound(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets, point = data
        images, targets, point = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float(), point.cuda(non_blocking=True).float()
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            P2, point_maps_pre, point_maps_pre1, point_maps_pre2 = model(images)
            point_loss = 0.0
            point3 = F.max_pool2d(point, (32, 32), (32, 32))
            point2 = F.max_pool2d(point, (16, 16), (16, 16))
            point1 = F.max_pool2d(point, (8, 8), (8, 8))

            for point_pre, point_pre1, point_pre2 in zip(
                    point_maps_pre, point_maps_pre1, point_maps_pre2):
                point_loss = point_loss + criterion(
                    point_pre, point1) + criterion(
                    point_pre1, point2) + criterion(point_pre2, point3)
            point_loss = point_loss / (3 * len(point_maps_pre1))
            seg_loss = 0.0

            for p in P2:
                seg_loss = seg_loss + structure_loss(p, targets)
            seg_loss = seg_loss / len(P2)
            loss = seg_loss + point_loss

            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()


def val_one_epoch_xbound(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader, ncols=70):
            img, msk, point = data
            img, msk, point = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float(), point.cuda(non_blocking=True).float()
            out, point_maps_pre, point_maps_pre1, point_maps_pre2 = model(
                img)
            out = torch.sigmoid(out)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch_xbound(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, ncols=70)):
            img, msk, point = data
            img, msk, point = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float(), point.cuda(
                non_blocking=True).float()
            out, point_maps_pre, point_maps_pre1, point_maps_pre2 = model(
                img)
            out = torch.sigmoid(out)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()

            preds.append(out)
            if i % config.save_interval == 0:
                # save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
                # ### use when predictions ####
                save_imgs(img, msk, out, i, config.work_dir, config.datasets, config.threshold, test_data_name=test_data_name)
            #     #### use when save imgs ####
                save_msk_pred(out, i, config.work_dir, config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


# MSCANet
def train_one_epoch_MSCANet(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out, _ = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()


def val_one_epoch_MSCANet(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader, ncols=70):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out, _ = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch_MSCANet(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, ncols=70)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out, _ = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()

            preds.append(out)
            if i % config.save_interval == 0:
                #     # save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
                #### use when predictions ####
                save_imgs(img, msk, out, i, config.work_dir, config.datasets, config.threshold,
                          test_data_name=test_data_name)
                #### use when save imgs ####
                save_msk_pred(out, i, config.work_dir, config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)

def train_one_epoch_multi_Ablation(train_loader,
                          model1,
                          model2,
                          criterion,
                          optimizer,
                          scheduler,
                          epoch,
                          logger,
                          config,
                          scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    coefficient_generator = nn.Sigmoid()
    model1.train()
    model2.train()

    loss_list = []
    alpha = coefficient_generator(torch.tensor(epoch / 100 * 20 - 10))
    # alpha = 1
    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, msk_seg, msk_contour = data
        images, msk_seg, msk_contour = images.cuda(non_blocking=True).float(), msk_seg.cuda(non_blocking=True).float(), msk_contour.cuda(non_blocking=True).float()
        msk_contour = msk_contour.squeeze(dim=1)
        if config.amp:
            with autocast():
                pred_seg = model1(images)
                pred_contour = model2(images)
                loss = Caculate_multi_task_loss(pred_seg, pred_contour, msk_seg, msk_contour, alpha)
                # loss = criterion(out, msk_seg)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_seg = model1(images)
            pred_contour = model2(images)
            loss = Caculate_multi_task_loss(pred_seg, pred_contour, msk_seg, msk_contour, alpha)
            # loss = criterion(out, msk_seg)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()


def val_one_epoch_multi_Ablation(test_loader,
                        model1,
                        model2,
                        criterion,
                        epoch,
                        logger,
                        config):
    # switch to evaluate mode
    # model = model
    model1.eval()
    model2.eval()
    preds = []
    gts = []
    loss_list = []
    alpha = 1
    # alpha = 0
    # coefficient_generator = nn.Sigmoid()
    # alpha = coefficient_generator(torch.tensor(epoch / 100 * 20 - 15))
    # alpha = coefficient_generator(torch.tensor(epoch / 100 * 20 - 5))
    with torch.no_grad():
        for data in tqdm(test_loader, ncols=70):
            img, msk_seg, msk_contour = data
            img, msk_seg, msk_contour = img.cuda(non_blocking=True).float(), msk_seg.cuda(non_blocking=True).float(), msk_contour.cuda(non_blocking=True).float()
            pred_seg = model1(img)
            pred_contour = model2(img)
            msk_contour = msk_contour.squeeze(dim=1)
            # loss = criterion(pred_seg, msk_seg)
            loss = Caculate_multi_task_loss(pred_seg, pred_contour, msk_seg, msk_contour, alpha)
            loss_list.append(loss.item())
            gts.append(msk_seg.squeeze(1).cpu().detach().numpy())
            if type(pred_seg) is tuple:
                pred_seg = pred_seg[0]
            pred_seg = pred_seg.squeeze(1).cpu().detach().numpy()
            preds.append(pred_seg)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch_multi_Ablation(test_loader,
                         model1,
                         model2,
                         criterion,
                         logger,
                         config,
                         test_data_name=None):
    # switch to evaluate mode
    model1.eval()
    model2.eval()
    preds = []
    gts = []
    loss_list = []
    # file = open("IouForImage.txt", "w")
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, ncols=70)):
            img, msk_seg, msk_contour = data
            img, msk_seg, msk_contour = img.cuda(non_blocking=True).float(), msk_seg.cuda(non_blocking=True).float(), msk_contour.cuda(non_blocking=True).float()
            pred_seg = model1(img)
            pred_contour = model2(img)
            loss = criterion(pred_seg, msk_seg)
            loss_list.append(loss.item())
            msk_seg = msk_seg.squeeze(1).cpu().detach().numpy()
            gts.append(msk_seg)
            if type(pred_seg) is tuple:
                pred_seg = pred_seg[0]
            pred_seg = pred_seg.squeeze(1).cpu().detach().numpy()
            pred_contour = torch.argmax(pred_contour, dim=1).squeeze(0).cpu().detach().numpy()
            # # calculate miou for every img
            # pred_seg = np.array(pred_seg).reshape(-1)
            # msk_seg = np.array(msk_seg).reshape(-1)
            # confusion = confusion_matrix(np.where(msk_seg >= 0.5, 1, 0), np.where(pred_seg >= config.threshold, 1, 0))
            # TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
            # miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
            # file.write("img_name = {}, miou = {}\n".format(i, miou))
            # # calculate miou for every img
            preds.append(pred_seg)
            if i % config.save_interval == 0:
                # save_imgs(img, msk_seg, pred_seg, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                #           test_data_name=test_data_name)
                # # output contains mask and contour
                # save_imgs_multitask(img, msk_seg, pred_seg, pred_contour, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                #           test_data_name=test_data_name)
                ##### use when prediction #####
                save_imgs_multitask(img, msk_seg, pred_seg, pred_contour, i, config.work_dir,
                                    config.datasets, config.threshold,
                                    test_data_name=test_data_name)
                ##### use when save mask #####
                save_msk_pred(pred_seg, i, config.work_dir, config.datasets, config.threshold, test_data_name=test_data_name)
                save_msk_contour(pred_contour, i, config.work_dir, config.datasets, config.threshold, test_data_name=test_data_name)

        # file.close()
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)