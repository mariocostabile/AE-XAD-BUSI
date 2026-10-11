# AE-XAD repo

This repository contains the code of the AE-XAD method (authored by Fabrizio Angiulli, Fabio Fassetti, Luca Ferragina and Simona Nisticò). If you use this code, please consider citing us!

```
@article{angiulli2025explaining,
  title={Explaining anomalies through semi-supervised Autoencoders},
  author={Angiulli, Fabrizio and Fassetti, Fabio and Ferragina, Luca and Nistic{\`o}, Simona},
  journal={Array},
  pages={100537},
  year={2025},
  publisher={Elsevier}
}
```

## Dataset
Dataset classes are used to load the train and test datasets for model training and evaluation. In the dataset package three classes are contained:
  * **CustomAD**, recommended for images of dimensions of up to $448 \times 448$, in this class no resize is applied to images
  * **MvtecAD**, recommended for images greater than $448 \times 448$, in this class the images are reshaped to $448 \times 448$
  * **CustomVGGAD**, recommended for images of dimensions of up to $448 \times 448$, in this class no resize is applied to images. It is mandatory to use this class when VGG or ResNet-based features extractors are used in the architectures.

Both classes accept channel-first RGB images, images, labels and ground truth that need to be stored in ndarrays saved in the same path. 
An example of how to arrange data can be found in **tools.create_dataset.mvtec** while the method **tools.create_dataset.mvtec** provides a function that can work with every dataset organized according to the following folder structure:

```
path
│   
└───train
│   │
│   └───good
│       │   img1.jpg(/.png/.npy)
│       │   img2.jpg(/.png/.npy)
│       │   ...
│   
└───test
│   │
│   └───good
│       │   img1.jpg(/.png/.npy)
│       │   img2.jpg(/.png/.npy)
│       │   ...
│   │
│   └───anomaly_type_1
│       │   img1.jpg(/.png/.npy)
│       │   img2.jpg(/.png/.npy)
│       │   ...
|   ...
│   │
│   └───anomaly_type_n
│       │   img1.jpg(/.png/.npy)
│       │   img2.jpg(/.png/.npy)
│       │   ...
│ 
└───ground_truth
│   │
│   └───anomaly_type_1
│       │   img1.jpg(/.png/.npy)
│       │   img2.jpg(/.png/.npy)
│       │   ...
|   ...
│   │
│   └───anomaly_type_n
│       │   img1.jpg(/.png/.npy)
│       │   img2.jpg(/.png/.npy)
│       │   ...

```

To insert a new dataset class it is sufficient to include a new class in the dataset package extending **torch.utils.data.Dataset**

## Network
Different network architectures are available:
 *  **shallow** - both the encoder and the decoder are one-layered networks. It works with flattened input which is mapped to a space of dimension *flat_dim*. Requires: *dim*, *flat_dim*, *latent_dim*
 *  **deep** - both the encoder and the decoder are two-layered networks. It works with flattened input which is mapped to a space of dimension *flat_dim*. Requires: *dim*, *flat_dim*, *intermediate_dim*, *latent_dim*
 *  **conv** - convolutional neural network with two image scale-down steps. Works with $(3 \times H \times W)$ or $(1 \times H \times W)$ images. Requires: *dim*
 *  **conv_deep** - convolutional neural network with two image scale-down steps. Differently from the latter one, it uses $(5 \times 5)$ filters instead of $(3 \times 3)$. Works with $(3 \times H \times W)$ or $(1 \times H \times W)$ images. Requires: *dim*
 *  **conv_deep_v2** - convolutional neural network with two image scale-down steps. Differently from the latter, the scale-down and scale-up operations are performed through convolutions and transposed convolution, respectively. Works with $(3 \times H \times W)$ or $(1 \times H \times W)$ images. Requires: *dim*
 *  **pca** - As **shallow** but without bias. Requires: *dim*, *flat_dim*, *latent_dim*


*dim* represents the shape of the $(3 \times H \times W)$ or $(1 \times H \times W)$ images.
*flat_dim* represents the flattenized dimension of the $(3 \times H \times W)$ or $(1 \times H \times W)$ images which, respectively, results to be $3 * H * W$ or $H * W$.
*latent_dim* represents the dimension of the latent space.
*intermediate_dim* intermediate dimention between *flat_dim* and *latent_dim*. it is recommended that its value is major than *latent_dim* and minor then *flat_dim*.

The recommended network for the mvtec dataset is the **conv_deep** one, while, for btad data, the **conv_deep_v2** network has been used. The results for the datasets belonging to the btad collection (referred as btad_01, btad_02 and btad_03) are available at the following link:  https://drive.google.com/drive/folders/1yx4Vh1z3CWGlaxBjMNXG4mnQS6pf8I4w?usp=sharing

## Trainer

The **Trainer** class is devoted to taking care of initializing and network and carrying out its training. It allows also to extracts the heatmaps, the outlierness scores and the reconstructed images of the considered test set.
To perform the initialization the following code can be used:

```
trainer = Trainer(latent_dim, lambda_p, lambda_s, f, path, AE_type, batch_size=None, silent=False, use_cuda=True,
                 loss='aexad', save_intermediate=False, dataset='mvtec')
```

where:
 * *lambda_p* is the weight of the anomalous pixels, when equal to None, is initialized according to the number of anomalous pixels
 * *lambda_s* is the weight of the anomalous samples, when equal to None, is initialized according to the number of anomalous samples
 * *f* is the function used to map the anomalous samples, the recommended one is `lambda x: torch.where(x>0.5, 0., 1.)`
 * *path* is the path in which the .npy files of the dataset are stored
 * *AE_type* is the type of architecture to use
 * *batch_size* is the number of samples to include into a batch, if None it is set equal to $1$, defaults to None
 * *silent* if true, training information are not displayed, defaults to False
 * *use_cuda* if True, the network uses the cuda memory for the training
 * *loss* the loss to use, possible values are 'aexad' and 'mse', it defaults to 'aexad'
 * *save_intermediate* if True, it saves model weights every $100$ epochs
 * *dataset* the name of the dataset to use, defaults to 'mvtec'

It follows an example with the recommended parameters:

```
trainer = Trainer(latent_dim=None, lambda_p=None, lambda_s=None, f=lambda x: torch.where(x>0.5, 0., 1.), path=path, AE_type='conv_deep', batch_size=16, silent=False, use_cuda=True,
                 loss='aexad', save_intermediate=False, dataset='mvtec')
```

