import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import albumentations as A


class CustomVGGAD_CP(Dataset):
    def __init__(self, path, train=True, img_size=224):
        super(CustomVGGAD_CP).__init__()
        self.img_size = img_size
        self.train = train
        #self.transform = self.transform_train() if self.train else self.transform_test()

        if self.train:
            split = 'train'
        else:
            split = 'test'

        x = np.load(os.path.join(path, f'X_{split}.npy')).swapaxes(1, 2).swapaxes(2, 3).astype(np.uint8)#[:,:,:,0]
        y = np.load(os.path.join(path, f'Y_{split}.npy'))
        gt = np.load((os.path.join(path, f'GT_{split}.npy'))) #/ 255.0).astype(np.float32)


        #normal_data = x[y == 0]
        #outlier_data = x[y == 1]

        #self.gt = gt
        self.gt = gt[:, 0].astype(np.uint8)
        print(gt.shape)
        self.labels = y
        self.images = x

        self.n_norm = int((1 - self.labels).sum())
        self.n_anom = int(self.labels.sum())

        self.reuse_times = 5
        self.repeat_time = 10
        self.anomaly_idx = np.append(np.argwhere(self.labels == 1).reshape(-1), np.arange(self.n_norm+self.n_anom, self.n_norm+self.n_anom*(self.reuse_times+self.repeat_time), 1), axis=0)
        self.normal_idx = np.argwhere(self.labels == 0).reshape(-1)

        #if self.train:
        #    self.images = np.append([self.images, np.full(((self.reuse_times-1)*self.n_anom, self.images.shape[1], self.images.shape[2], self.images.shape[3]),
        #                                                  fill_value=self.images[self.labels==1])],
        #                            axis=0)
        #    self.labels = np.append(self.labels, np.ones(self.n_anom*(self.repeat_time + self.reuse_times-1)), axis=0)
        #print(self.images.shape)

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        #self.mean = (x.reshape((-1, 3)) / 255.).mean(axis=0)
        #self.std = (x.reshape((-1, 3)) / 255.).std(axis=0)

        self.dim = (self.images.shape[3], self.images.shape[1], self.images.shape[2])

        test_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            #transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        train_transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomChoice([
                transforms.ColorJitter(0.04, 0.04, 0.04, 0.04),
                transforms.ColorJitter(0.005, 0.0005, 0.0005, 0.0005),
            ]),
            # transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        # Aggiunta 30/06/25
        self.augmentors = [  A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OpticalDistortion(p=1.0, distort_limit=1.0),
            A.OneOf([
                A.GaussNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.ElasticTransform(p=0.3)(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3)]

        self.transform_mask = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor()])

        # ------------------------------------------------

        if self.train:
            self.transform = train_transform
        else:
            self.transform = test_transform

#
    #def transform_test(self):
    #    composed_transforms = transforms.Compose([
    #        transforms.Resize((self.args.img_size, self.args.img_size)),
    #        transforms.ToTensor(),
    #        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #    return composed_transforms
    ## -----------------------------------------

    def real_dim(self):
        return self.n_norm + self.n_anom
    def __len__(self):
        if self.train:
            return self.n_norm + self.repeat_time * self.n_anom + self.reuse_times * self.n_anom
        else:
            return self.n_norm + self.n_anom
    def __getitem__(self, index):
        #transform = self.transform
        #image = Image.fromarray(self.images[index])
        #image = self.images[index]
        ##image = self.transform(image)
        #sample = {'image': transform(image), 'label': self.labels[index]}
        #if index in self.outlier_idx:
        #    image_label = self.gt[index-self.out_idx_start]
        #else:
        #    image_label = np.zeros_like(self.images[index])
        #image_label = self.gt[index] #/ 255.

        if self.train and index >= self.n_norm + self.n_anom:
            _index = self.anomaly_idx[(index - self.n_norm - self.n_anom) % self.n_anom]
            image = self.images[_index]
            image_label = self.gt[_index]
            if (index - self.n_norm - self.n_anom) // self.n_anom >= self.reuse_times:
                aug = self.randAugmenter()
                #mask in [0,255]
                #augmentated = aug(image=image, mask=image_label)
                image, image_label = self.copy_paste(image, image_label)
                image, image_label = Image.fromarray(image), Image.fromarray(image_label)
                #image, image_label = augmentated['image'], augmentated['mask']
            else:
                image, image_label = Image.fromarray(image), Image.fromarray(image_label)
            label = 1
        else:
            image = Image.fromarray(self.images[index])
            image_label = Image.fromarray(self.gt[index])
            label = self.labels[index]

        image_label = self.transform_mask(image_label).repeat(3, 1, 1)

        sample = {'image':  self.transform(image), 'label': label, 'gt_label': image_label}
        return sample


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmentors)), 3, replace=False)
        aug = A.Compose([self.augmentors[aug_ind[0]],
                         self.augmentors[aug_ind[1]],
                         self.augmentors[aug_ind[2]]])
        # aug = A.Compose([self.augmentors[0], A.GridDistortion(p=1.0), self.augmentors[3], self.augmentors[1], self.augmentors[7]])
        return aug

    #def getitem(self, index):
    #    # if index in self.outlier_idx and self.train:
    #    #     transform = self.transform_anomaly
    #    # else:
    #    #     transform = self.transform
    #    #image = Image.fromarray(self.images[index])
    #    image = self.images[index]
    #    if index in self.outlier_idx:
    #        image_label = self.gt[index-self.out_idx_start]
    #    else:
    #        image_label = np.zeros_like(self.images[index])
    #    sample = {'image': image, 'label': self.labels[index], 'gt_label': image_label}
    #    return sample

    def copy_paste(self, img, mask):
        n_idx = np.random.randint(self.n_norm)  # get a random normal sample
        aug = self.randAugmenter()

        #image = cv2.imread(img)  # anomaly sample
        image = img
        # temp_img = Image.open(img)
        # temp_img.save("aug_imgs/ano_img.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (900, 900, 3)
        #n_image = cv2.imread(self.n_imgs[n_idx])  # normal sample
        n_image = self.images[self.labels==0][n_idx]
        n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB)  # (900, 900, 3)
        # temp = Image.open(self.n_imgs[n_idx])
        # temp.save('aug_imgs/nor_img.jpg')
        img_height, img_width = n_image.shape[0], n_image.shape[1]

        #mask = Image.open(mask)
        #mask = np.asarray(mask)  # (900, 900)

        # augment the abnormal region
        augmentated = aug(image=image, mask=mask)
        aug_image, aug_mask = augmentated['image'], augmentated['mask']
        # temp_img = Image.fromarray(aug_image)
        # temp_img.save("aug_imgs/ano_aug_img.jpg")
        # crop_img = aug_image.copy()
        # crop_img[aug_mask == 0] = 0
        # crop_img = Image.fromarray(crop_img)
        # crop_img.save('aug_imgs/crop_img.jpg')
        self.in_fg_region = False
        if self.in_fg_region:
            n_img_path = self.n_imgs[n_idx]
            img_file = n_img_path.split('/')[-1]
            fg_path = os.path.join(f'fg_mask/{self.class_name}', img_file)
            fg_mask = Image.open(fg_path)
            #fg_mask.save("aug_imgs/fg_mask.jpg")
            fg_mask = np.asarray(fg_mask)

            intersect_mask = np.logical_and(fg_mask == 255, aug_mask == 255)
            if (np.sum(intersect_mask) > int(2 / 3 * np.sum(aug_mask == 255))):
                # when most part of aug_mask is in the fg_mask region
                # copy the augmentated anomaly area to the normal image
                n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                return n_image, aug_mask
            else:
                contours, _ = cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                center_xs, center_ys = [], []
                widths, heights = [], []
                for i in range(len(contours)):
                    M = cv2.moments(contours[i])
                    if M['m00'] == 0:  # error case
                        x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                        y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                        center_x = int((x_min + x_max) / 2)
                        center_y = int((y_min + y_max) / 2)
                    else:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                    center_xs.append(center_x)
                    center_ys.append(center_y)
                    x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                    y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                    width, height = x_max - x_min, y_max - y_min
                    widths.append(width)
                    heights.append(height)
                if len(widths) == 0 or len(heights) == 0:  # no contours
                    n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                    return n_image, aug_mask
                else:
                    max_width, max_height = np.max(widths), np.max(heights)
                    center_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    center_mask[int(max_height/2):img_height-int(max_height/2), int(max_width/2):img_width-int(max_width/2)] = 255
                    fg_mask = np.logical_and(fg_mask == 255, center_mask == 255)

                    x_coord = np.arange(0, img_width)
                    y_coord = np.arange(0, img_height)
                    xx, yy = np.meshgrid(x_coord, y_coord)
                    # coordinates of fg region points
                    xx_fg = xx[fg_mask]
                    yy_fg = yy[fg_mask]
                    xx_yy_fg = np.stack([xx_fg, yy_fg], axis=-1)  # (N, 2)

                    if xx_yy_fg.shape[0] == 0:  # no fg
                        n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                        return n_image, aug_mask

                    aug_mask_shifted = np.zeros((img_height, img_width), dtype=np.uint8)
                    for i in range(len(contours)):
                        aug_mask_shifted_i = np.zeros((img_height, img_width), dtype=np.uint8)
                        new_aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                        # random generate a point in the fg region
                        idx = np.random.choice(np.arange(xx_yy_fg.shape[0]), 1)
                        rand_xy = xx_yy_fg[idx]
                        delta_x, delta_y = center_xs[i] - rand_xy[0, 0], center_ys[i] - rand_xy[0, 1]

                        x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                        y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])

                        # mask for one anomaly region
                        aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                        aug_mask_i[y_min:y_max, x_min:x_max] = 255
                        aug_mask_i = np.logical_and(aug_mask == 255, aug_mask_i == 255)

                        # coordinates of orginal mask points
                        xx_ano, yy_ano = xx[aug_mask_i], yy[aug_mask_i]

                        # shift the original mask into fg region
                        xx_ano_shifted = xx_ano - delta_x
                        yy_ano_shifted = yy_ano - delta_y
                        outer_points_x = np.logical_or(xx_ano_shifted < 0, xx_ano_shifted >= img_width)
                        outer_points_y = np.logical_or(yy_ano_shifted < 0, yy_ano_shifted >= img_height)
                        outer_points = np.logical_or(outer_points_x, outer_points_y)

                        # keep points in image
                        xx_ano_shifted = xx_ano_shifted[~outer_points]
                        yy_ano_shifted = yy_ano_shifted[~outer_points]
                        aug_mask_shifted_i[yy_ano_shifted, xx_ano_shifted] = 255

                        # original points should be changed
                        xx_ano = xx_ano[~outer_points]
                        yy_ano = yy_ano[~outer_points]
                        new_aug_mask_i[yy_ano, xx_ano] = 255
                        # copy the augmentated anomaly area to the normal image
                        n_image[aug_mask_shifted_i == 255, :] = aug_image[new_aug_mask_i == 255, :]
                        aug_mask_shifted[aug_mask_shifted_i == 255] = 255
                    return n_image, aug_mask_shifted
        else:  # no fg restriction
            # copy the augmentated anomaly area to the normal image
            n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]

            return n_image, aug_mask


class CustomVGGAD(Dataset):
    def __init__(self, path, train=True, img_size=224):
        super(CustomVGGAD_CP).__init__()
        self.img_size = img_size
        self.train = train
        #self.transform = self.transform_train() if self.train else self.transform_test()

        if self.train:
            split = 'train'
        else:
            split = 'test'

        x = np.load(os.path.join(path, f'X_{split}.npy')).swapaxes(1, 2).swapaxes(2, 3).astype(np.uint8)#[:,:,:,0]
        y = np.load(os.path.join(path, f'Y_{split}.npy'))
        gt = np.load((os.path.join(path, f'GT_{split}.npy'))).swapaxes(1, 2).swapaxes(2, 3).astype(np.uint8)


        #normal_data = x[y == 0]
        #outlier_data = x[y == 1]

        self.gt = gt
        self.labels = y
        self.images = x

        self.n_norm = int((1 - self.labels).sum())
        self.n_anom = int(self.labels.sum())

        self.anomaly_idx = np.argwhere(self.labels == 1).reshape(-1)
        self.normal_idx = np.argwhere(self.labels == 0).reshape(-1)


        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        #self.mean = (x.reshape((-1, 3)) / 255.).mean(axis=0)
        #self.std = (x.reshape((-1, 3)) / 255.).std(axis=0)

        self.dim = (self.images.shape[3], self.images.shape[1], self.images.shape[2])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            #transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomChoice([
                transforms.ColorJitter(0.04, 0.04, 0.04, 0.04),
                transforms.ColorJitter(0.005, 0.0005, 0.0005, 0.0005),
            ]),
            # transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])


        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])

        # ------------------------------------------------

        if self.train:
            self.transform = train_transform
        else:
            self.transform = test_transform

#
    def __len__(self):
        return self.n_norm + self.n_anom
    def __getitem__(self, index):
        image_label = self.gt[index] #/ 255.
        label = self.labels[index]
        image = self.images[index]

        sample = {'image': self.transform(image), 'label': label, 'gt_label':  self.transform_mask(image_label)}
        return sample



class CustomVGGAD_AE(Dataset):
    def __init__(self, path, train=True, img_size=224):
        super(CustomVGGAD_CP).__init__()
        self.train = train
        #self.transform = self.transform_train() if self.train else self.transform_test()

        if self.train:
            split = 'train'
        else:
            split = 'test'

        x = np.load(os.path.join(path, f'X_{split}.npy')).swapaxes(1, 2).swapaxes(2, 3).astype(np.uint8) #/ 255.0)[:,:,:,0]
        y = np.load(os.path.join(path, f'Y_{split}.npy'))

        self.labels = y
        self.images = x[y==0]

        self.img_size = img_size

        self.dim = self.images.shape[1:]

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])



        # self.mean = (x.reshape((-1, 3)) / 255.).mean(axis=0)
        # self.std = (x.reshape((-1, 3)) / 255.).std(axis=0)

        self.dim = (self.images.shape[3], self.images.shape[1], self.images.shape[2])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            # transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomChoice([
                transforms.ColorJitter(0.04, 0.04, 0.04, 0.04),
                transforms.ColorJitter(0.005, 0.0005, 0.0005, 0.0005),
            ]),
            # transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        self.transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()])

        # ------------------------------------------------

        if self.train:
            self.transform = train_transform
        else:
            self.transform = test_transform


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        #image = Image.fromarray(self.images[index], mode='RGB')
        image = self.images[index]
        image = self.transform(image)
        sample = {'image': image}
        return sample


