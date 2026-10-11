import os

import matplotlib.pyplot as plt
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from time import time

from AE_architectures import Shallow_Autoencoder, Deep_Autoencoder, Conv_Autoencoder, PCA_Autoencoder, Conv_Deep_Autoencoder, Conv_Deep_Autoencoder_v2, ResNet_CNN_Attn, ResNet_CNN_Attn_ablation
from dataloaders.custom import CustomAD, CustomAD_AE 
from dataloaders.mvtec import MvtecAD
from dataloaders.custom_VGG import CustomVGGAD_CP, CustomVGGAD, CustomVGGAD_AE

from loss import AEXAD_loss, AEXAD_loss_weighted, AEXAD_loss_norm, AEXAD_loss_norm_vgg, MSE_loss_vgg
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self,
                 steps_per_epoch,
                 batch_size,
                 dataset):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.dataset = dataset

        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size


        self.normal_generator = self.randomGenerator(self.dataset.normal_idx)
        self.outlier_generator = self.randomGenerator(self.dataset.anomaly_idx)
        # n_normal: 2/3; n_outlier: 1/3
        self.n_normal = 2 * self.batch_size // 3
        self.n_anomaly = self.batch_size - self.n_normal

    def randomGenerator(self, list):
        while True:
            random_list = np.random.permutation(list)
            for i in random_list:
                yield i

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_anomaly):
                batch.append(next(self.outlier_generator))

            yield batch
class Trainer:
    def __init__(self, latent_dim, lambda_p, lambda_s, f, path, AE_type, batch_size=None, silent=False, use_cuda=True,
                 loss='aexad', save_intermediate=False, dataset='mnist', steps_per_epoch=20, only_anom=False):
        '''
        :param latent_dim:
        :param lambda_p: float, anomalous pixel weight, if none the value is inferred from the dataset
        :param lambda_s: float, anomalous samples weight, if none the value is inferred from the dataset
        :param f: func, mapping function for anomalous pixels
        :param path: str path in which checkpoints are saved
        :param AE_type: str Type of architecture considered. Possible values are: shallow, deep, conv, conv_deep, conv_f2, vgg_cnn, pca
        :param batch_size: int number of samples composing a batch, defaults to None
        :param silent: bool, if True, deactivate progress bar, defaults to False
        :param use_cuda: bool, if True the network uses gpu, defaults to True
        :param loss: str, loss to use, possible values are aexad and mse, defaults to aexad
        :param save_intermediate: bool, if True save a checkpoint of the model every 100 epochs, defaults to False
        :param dataset: str, dataset to use, default to mnist, for custom dataset type the dataset name
        '''
        self.silent = silent
        
        if dataset == 'mnist' or dataset == 'fmnist':
            ds = CustomAD(path, train=True)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(ds.labels))
            self.train_loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
            self.test_loader = DataLoader(CustomAD(path, train=False), batch_size=batch_size, shuffle=False)

        elif (dataset == 'tf_ds' or dataset =='mvtec') and 'vgg_cnn' in AE_type:
            if loss == 'mse' and not only_anom:
                ds = CustomVGGAD_AE(path, train=True)
                self.train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
            else:
                if only_anom:
                    ds = CustomVGGAD(path, train=True)
                    self.train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
                else:
                    ds = CustomVGGAD_CP(path, train=True)
                    self.train_loader = DataLoader(ds, batch_sampler=BalancedBatchSampler(steps_per_epoch, batch_size, ds))
            self.test_loader = DataLoader(CustomVGGAD(path, train=False), batch_size=1, shuffle=False)
        else:
            ds = MvtecAD(path, train=True)
            weights = np.where(ds.labels == 1, 0.6, 0.4)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(ds.labels))
            self.train_loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
            self.test_loader = DataLoader(MvtecAD(path, train=False), batch_size=batch_size, shuffle=False)

        self.save_intermediate = save_intermediate

        if lambda_s is None:
            lambda_s = len(self.train_loader.dataset) / np.sum(self.train_loader.dataset.labels)
            print(lambda_s)

        self.cuda = use_cuda and torch.cuda.is_available()

        if AE_type == 'shallow':
            self.model = Shallow_Autoencoder(self.train_loader.dataset.dim, np.prod(self.train_loader.dataset.dim),
                                             latent_dim)
        # deep
        elif AE_type == 'deep':
            self.model = Deep_Autoencoder(self.train_loader.dataset.dim, flat_dim=np.prod(self.train_loader.dataset.dim),
                                          intermediate_dim=256, latent_dim=latent_dim)
        # conv
        elif AE_type == 'conv':
            self.model = Conv_Autoencoder(self.train_loader.dataset.dim)

        elif AE_type == 'conv_deep':
            self.model = Conv_Deep_Autoencoder(self.train_loader.dataset.dim)

        elif AE_type == 'conv_deep_v2':
            self.model = Conv_Deep_Autoencoder_v2(self.train_loader.dataset.dim)

        elif AE_type == 'vgg_cnn_abl':
            assert (isinstance(self.train_loader.dataset, CustomVGGAD_CP) or isinstance(self.train_loader.dataset, CustomVGGAD)
                    or isinstance(self.train_loader.dataset, CustomVGGAD_AE))
            self.model = ResNet_CNN_Attn_ablation(self.train_loader.dataset.dim)

        elif 'vgg_cnn' in AE_type:
            assert (isinstance(self.train_loader.dataset, CustomVGGAD_CP) or isinstance(self.train_loader.dataset, CustomVGGAD) or isinstance(self.train_loader.dataset, CustomVGGAD_AE))
            self.model = ResNet_CNN_Attn(self.train_loader.dataset.dim)

        elif AE_type == 'pca':
            self.model = PCA_Autoencoder(np.prod(self.train_loader.dataset.dim), np.prod(self.train_loader.dataset.dim),
                                         latent_dim)
        else:
            raise Exception('Model not yet implemented')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)


        self.loss = loss
        if loss == 'aexad':
            if isinstance(self.train_loader.dataset, CustomVGGAD_CP) or isinstance(self.train_loader.dataset, CustomVGGAD)\
                    or isinstance(self.train_loader.dataset, CustomVGGAD_AE):
                self.criterion = AEXAD_loss_norm_vgg(self.train_loader.dataset.mean,
                                 self.train_loader.dataset.std, lambda_p, lambda_s, f, self.cuda)
            else:
                self.criterion = AEXAD_loss_norm(lambda_p, lambda_s, f, self.cuda)
            #self.criterion = AEXAD_loss(lambda_p, lambda_s, f, self.cuda)
        elif loss == 'mse':
            if isinstance(self.train_loader.dataset, CustomVGGAD_CP):
                self.criterion = MSE_loss_vgg(self.train_loader.dataset.mean, self.train_loader.dataset.std)
            else:
                self.criterion = nn.MSELoss()

        if self.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def reconstruct(self):
        '''
        Reconstruct the images of the test data loader
        '''
        self.model.eval()
        tbar = tqdm(self.test_loader, disable=self.silent)
        shape = [0]
        shape.extend(self.test_loader.dataset.images.shape[1:])
        rec_images = []
        for i, sample in enumerate(tbar):
            image, label, gtmap = sample['image'], sample['label'], sample['gt_label']
            if self.cuda:
                image = image.cuda()
            output = self.model(image).detach().cpu().numpy()
            rec_images.extend(output)
        rec_images = np.array(rec_images)
        rec_images = rec_images.swapaxes(1, 2).swapaxes(2, 3)
        if isinstance(self.train_loader.dataset, CustomVGGAD_CP):
            rec_images = rec_images * self.train_loader.dataset.std + self.train_loader.dataset.std
        return rec_images


    def test(self):
        '''
        Test the model on the test set provided by the test data loader
        '''
        tbar = tqdm(self.test_loader, disable=self.silent)
        shape = [0]
        shape.extend(self.test_loader.dataset.images.shape[1:])
        heatmaps = []
        scores = []
        gtmaps = []
        labels = []

        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                image, label, gtmap = sample['image'], sample['label'], sample['gt_label']
                if self.cuda:
                    image = image.cuda()
                    output = self.model(image).detach().cpu().numpy()
                image = image.cpu().numpy()
                if isinstance(self.train_loader.dataset, CustomVGGAD_CP) or isinstance(self.train_loader.dataset, CustomVGGAD) \
                    or isinstance(self.train_loader.dataset, CustomVGGAD_AE):
                    #print('Testing on vgg')
                    mean = self.test_loader.dataset.mean[np.newaxis, :, np.newaxis, np.newaxis]
                    std = self.test_loader.dataset.std[np.newaxis, :, np.newaxis, np.newaxis]
                    #image = (image.reshape((image.shape[0], image.shape[1], -1)) * std + mean).reshape(image.shape)
                    image = image * std + mean
                heatmap = ((image-output) ** 2)#.numpy() np.abs(image-output)
                score = heatmap.reshape((image.shape[0], -1)).mean(axis=-1)
                heatmaps.extend(heatmap)
                scores.extend(score)
                gtmaps.extend(gtmap.detach().numpy())
                labels.extend(label.detach().numpy())
                #plt.figure()
                #plt.subplot(1, 3, 1)
                #plt.imshow(image[0].swapaxes(0, 1).swapaxes(1, 2))
                #plt.subplot(1, 3, 2)
                #plt.imshow(output[0].swapaxes(0, 1).swapaxes(1, 2))
                #plt.subplot(1, 3, 3)
                #plt.imshow(gtmap[0].numpy().swapaxes(0, 1).swapaxes(1, 2))
                #plt.savefig(f'test_{i}_{self.loss}.jpg')
                ##plt.close('all')
            scores = np.array(scores)
            heatmaps = np.array(heatmaps)
            gtmaps = np.array(gtmaps)
            labels = np.array(labels)
        return heatmaps, scores, gtmaps, labels


    def stop_fe_training(self):
        assert isinstance(self.model, VGG_CNN_mask)

        for param in self.model.encoder_pre.parameters():
            param.requires_grad = False


    def train(self, epochs, save_path='.'):
        '''
        Trains the model on the train set provided by the train data loader
        :param epochs: int, number of epochs
        :param save_path: str, path used for saving the model, defaults to '.'
        '''
        if isinstance(self.model, Conv_Autoencoder):
            name = 'model_conv'
        elif isinstance(self.model, Deep_Autoencoder):
            name = 'model_deep'
        elif isinstance(self.model, Shallow_Autoencoder):
            name = 'model'
        elif isinstance(self.model, Conv_Deep_Autoencoder):
            name = 'model_conv_deep'
        elif isinstance(self.model, Conv_Deep_Autoencoder_v2):
            name = 'model_conv_deep_v2'
        elif isinstance(self.model, ResNet_CNN_Attn):
            name = 'model_vgg_cnn'
        elif isinstance(self.model, ResNet_CNN_Attn_ablation):
            name = 'model_vgg_cnn_abl'

        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            train_loss_n = 0.0
            train_loss_a = 0.0
            na = 0
            nn = 0
            ns = 0
            tbar = tqdm(self.train_loader, disable=self.silent)
            for i, sample in enumerate(tbar):
                if self.loss == 'mse':
                    image = sample['image']

                    if self.cuda:
                        image = image.cuda()
                    output = self.model(image)

                    loss = self.criterion(output, image)
                else:
                    image, label, gt_label = sample['image'], sample['label'], sample['gt_label']

                    if self.cuda:
                        image = image.cuda()
                        gt_label = gt_label.cuda()
                        label = label.cuda()
                    output = self.model(image)

                    loss, loss_n, loss_a = self.criterion(output, image, gt_label, label)
                    na += label.sum()
                    nn += image.shape[0] - na
                ns += 1
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                train_loss += loss.item()
                if not self.loss == 'mse':
                    train_loss_n += loss_n.item()
                    train_loss_a += loss_a.item()


                if self.loss == 'mse':
                    tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / ns))
                else:
                    tbar.set_description('Epoch:%d, Train loss: %.3f, Normal loss: %.3f, Anom loss: %3f' % (epoch, train_loss / ns, loss_n, loss_a))


        torch.save(self.model.state_dict(), os.path.join(save_path, f'{self.loss}_{name}.pt'))

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(filename)) #args.experiment_dir, filename))


def launch(data_path, epochs, batch_size, latent_dim, lambda_p, lambda_s, f, AE_type, loss='aexad',
           save_intermediate=False, save_path='', use_cuda=True, dataset='mnist', steps_per_epoch=20, only_anom=False):
    trainer = Trainer(latent_dim, lambda_p, lambda_s, f, data_path, AE_type, batch_size, loss=loss,
                      save_intermediate=save_intermediate, use_cuda=use_cuda, dataset=dataset,
                      steps_per_epoch=steps_per_epoch, only_anom=only_anom)


    start = time()
    trainer.train(epochs, save_path=save_path)
    tot_time = time() - start

    heatmaps, scores, gtmaps, labels = trainer.test()
    #torch.save(trainer.model.state_dict(), os.path.join(save_path, f'{loss}_conv_deep_v2.pt'))

    return heatmaps, scores, gtmaps, labels, tot_time
