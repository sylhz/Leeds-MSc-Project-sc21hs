'''
Some util functions
Part of the code is referenced from Kaggle
'''

import os
import cv2
import torch
import random
import numpy as np
from . import fmix
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.cuda.amp import autocast


def seed_everything(seed):
    '''Fixed various random seeds to facilitate ablation experiments
    Args:
        seed :  int
    '''
    # fixed random seed for scipy
    random.seed(seed)  # fixed random seed for random library
    os.environ['PYTHONHASHSEED'] = str(seed)  # fixed randomness of python hashes (not necessarily effective)
    np.random.seed(seed)  # fixed random seed for numpy
    torch.manual_seed(seed)  # fixed random seed for torch cpu computation
    torch.cuda.manual_seed(seed)  # fixed random seed for torch cuda computations
    torch.backends.cudnn.deterministic = True  # Whether to fix the calculation implementation of the convolution operator
    torch.backends.cudnn.benchmark = True  # Whether to enable automatic optimization, choose the fastest convolution calculation method


def get_img(path):
    '''Load the image using opencv
    For historical reasons, the image format read by opencv is bgr
    Args:
        path : str  image file path e.g '../data/train_img/1.jpg'
    '''
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def rand_bbox(size, lam):
    '''The bbox interception function of cutmix
    Args:
        size : tuple image size e.g (256,256)
        lam  : float intercept ratio
    Returns:
        upper left and lower right coordinates of bbox
        int,int,int,int
    '''
    W = size[0]  # Capture the width of the image
    H = size[1]  # Capture the height of the image
    cut_rat = np.sqrt(1. - lam)  # The bbox ratio to be clipped
    cut_w = np.int(W * cut_rat)  # width of bbox to be cut
    cut_h = np.int(H * cut_rat)  # height of bbox to be cut

    cx = np.random.randint(W)  # Uniformly distributed sampling, randomly select the x-coordinate of the center point of the intercepted bbox
    cy = np.random.randint(H)  # uniformly distributed sampling, randomly select the y-coordinate of the center point of the intercepted bbox

    bbx1 = np.clip(cx - cut_w // 2, 0, W)  # top left x coordinate
    bby1 = np.clip(cy - cut_h // 2, 0, H)  # upper left corner y coordinate
    bbx2 = np.clip(cx + cut_w // 2, 0, W)  # bottom right corner x coordinate
    bby2 = np.clip(cy + cut_h // 2, 0, H)  # bottom right y coordinate
    return bbx1, bby1, bbx2, bby2


class CassavaDataset(Dataset):
    '''Cassava Leaf Competition Data Loading Class
    Attributes:
        __len__ : The number of samples of the data.
        __getitem__ : Index function.
    '''
    def __init__(
            self,
            df,
            data_root,
            transforms=None,
            output_label=True,
            one_hot_label=False,
            do_fmix=False,
            fmix_params={
                'alpha': 1.,
                'decay_power': 3.,
                'shape': (512, 512),
                'max_soft': 0.3,
                'reformulate': False
            },
            do_cutmix=False,
            cutmix_params={
                'alpha': 1,
            }):
        '''
        Args:
             df : DataFrame , the filename and label of the sample image
             data_root : str , the file path where the image is located, absolute path
             transforms : object , image enhancement
             output_label : bool , whether to output the label
             one_hot_label : bool , whether to perform onehot encoding
             do_fmix : bool , whether to use fmix
             fmix_params :dict , fmix parameters {'alpha':1.,'decay_power':3.,'shape':(256,256),'max_soft':0.3,'reformulate':False}
             do_cutmix : bool, whether to use cutmix
             cutmix_params : dict , parameters of cutmix {'alpha':1.}
         Raises:

        '''
        super().__init__()
        self.df = df.reset_index(drop=True).copy()  # rebuild the index
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        if output_label:
            self.labels = self.df['label'].values
            if one_hot_label:
                self.labels = np.eye(self.df['label'].max() +
                                     1)[self.labels]  # Generate onehot encoding using identity matrix

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        '''
        Args:
            index : int , index
        Returns:
            img, target(optional)
        '''
        if self.output_label:
            target = self.labels[index]

        img = get_img(
            os.path.join(self.data_root,
                         self.df.loc[index]['image_id']))  # Splicing address, loading image

        if self.transforms:  # Use image augmentation
            img = self.transforms(image=img)['image']

        if self.do_fmix and np.random.uniform(
                0., 1., size=1)[0] > 0.5:  # 50% chance to trigger fmix data augmentation

            with torch.no_grad():
                lam, mask = sample_mask(
                    **self.fmix_params)  

                fmix_ix = np.random.choice(self.df.index,
                                           size=1)[0]  # Randomly select images to mix
                fmix_img = get_img(
                    os.path.join(self.data_root,
                                 self.df.loc[fmix_ix]['image_id']))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)

                img = mask_torch * img + (1. - mask_torch) * fmix_img  # mix image

                rate = mask.sum() / float(img.size)  # Get the rate of the mix
                target = rate * target + (
                    1. - rate) * self.labels[fmix_ix]  # target to mix

        if self.do_cutmix and np.random.uniform(
                0., 1., size=1)[0] > 0.5:  # 50% chance to trigger cutmix data augmentation
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img = get_img(
                    os.path.join(self.data_root,
                                 self.df.loc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']

                lam = np.clip(
                    np.random.beta(self.cutmix_params['alpha'],
                                   self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox(cmix_img.shape[:2], lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2,
                                                        bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) *
                            (bby2 - bby1) / float(img.size))  # Get the rate of the mix
                target = rate * target + (
                    1. - rate) * self.labels[cmix_ix]  # target to mix

        if self.output_label:
            return img, target
        else:
            return img


def prepare_dataloader(df, trn_idx, val_idx, data_root, trn_transform,
                       val_transform, bs, n_job):
    '''Multi-process data generator
     Args:
         df : DataFrame , the filename and label of the sample image
         trn_idx : ndarray , list of training set indices
         val_idx : ndarray , list of validation set indices
         data_root : str , the path where the image file is located
         trn_transform : object , training set image enhancer
         val_transform : object , validation set image enhancer
         bs : int , the number of batchsize each time
         n_job : int , the number of processes used
     Returns:
         train_loader, val_loader , data generators for training and validation sets
    '''
    train_ = df.loc[trn_idx, :].reset_index(drop=True)  # rebuild the index
    valid_ = df.loc[val_idx, :].reset_index(drop=True)  # rebuild the index

    train_ds = CassavaDataset(train_,
                              data_root,
                              transforms=trn_transform,
                              output_label=True,
                              one_hot_label=False,
                              do_fmix=False,
                              do_cutmix=False)
    valid_ds = CassavaDataset(valid_,
                              data_root,
                              transforms=val_transform,
                              output_label=True,
                              one_hot_label=False,
                              do_fmix=False,
                              do_cutmix=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=bs,
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=n_job,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=bs,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=n_job,
    )

    return train_loader, val_loader


def train_one_epoch(epoch,
                    model,
                    loss_fn,
                    optimizer,
                    train_loader,
                    device,
                    scaler,
                    scheduler=None,
                    schd_batch_update=False,
                    accum_iter=2):
    '''training set each epoch training function
     Args:
         epoch : int , which epoch to train to
         model : object, the model to train
         loss_fn : object, loss function
         optimizer : object, optimization method
         train_loader : object, training set data generator
         scaler : object, gradient amplifier
         device : str , the training device used e.g 'cuda:0'
         scheduler : object , learning rate adjustment policy
         schd_batch_update : bool, if it is true, each batch will be adjusted, otherwise it will be adjusted after an epoch ends
         accum_iter : int , gradient accumulation
     '''

    model.train()  # Turn on training mode

    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))  # Construct a progress bar

    for step, (imgs, image_labels) in pbar:  # iterate over each batch
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        with autocast():  # Turn on auto mix precision
            image_preds = model(imgs)  # Forward Propagation, Calculate Predicted Values
            loss = loss_fn(image_preds, image_labels)  # Calculate loss

        scaler.scale(loss).backward()  

        # loss regularization, using exponential averaging
        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * .99 + loss.item() * .01

        if ((step + 1) % accum_iter == 0) or ((step + 1) == len(train_loader)):
            scaler.step(
                optimizer)  # unscale gradient, if the gradient has no overflow, use opt to update the gradient, otherwise do not update
            scaler.update()  # wait for the next scale gradient
            optimizer.zero_grad()  # Gradient clear

            if scheduler is not None and schd_batch_update:  # Learning rate adjustment strategy
                scheduler.step()

        # 打印 loss 值
        description = f'epoch {epoch} loss: {running_loss:.4f}'
        pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:  # Learning rate adjustment strategy
        scheduler.step()


def valid_one_epoch(epoch, model, loss_fn, val_loader, device):
    '''Validation set inference
     Args:
         epoch : int, the number of epoch
         model : object, model
         loss_fn : object, loss function
         val_loader : object, validation set data generator
         device : str , the training device used e.g 'cuda:0'
     '''

    model.eval()  # Enable inference mode

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))  # Construct a progress bar

    for step, (imgs, image_labels) in pbar:  # iterate over each batch
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)  # forward Propagation, calculate Predicted Values
        image_preds_all += [
            torch.argmax(image_preds, 1).detach().cpu().numpy()
        ]  # Get predicted labels
        image_targets_all += [image_labels.detach().cpu().numpy()]  # Get real labels

        loss = loss_fn(image_preds, image_labels)  # Calculate the loss

        loss_sum += loss.item() * image_labels.shape[0]  # Calculate the sum of loss
        sample_num += image_labels.shape[0]  # Number of samples

        description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'  # print average loss
        pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format(
        (image_preds_all == image_targets_all).mean()))  # print accuracy


if __name__ == '__main__':
    pass
