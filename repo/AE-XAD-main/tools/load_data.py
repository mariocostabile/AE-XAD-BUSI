import numpy as np
import torch

from AE_architectures import Shallow_Autoencoder, Deep_Autoencoder, Conv_Autoencoder, Conv_Deep_Autoencoder, \
    PCA_Autoencoder, Conv_Deep_Autoencoder_v2


def load_model(path, dim, latent_dim=None, AE_type='conv_deep'):
    if AE_type == 'shallow':
        model = Shallow_Autoencoder(dim, np.prod(dim), latent_dim)
    # deep
    elif AE_type == 'deep':
        model = Deep_Autoencoder(dim, flat_dim=np.prod(dim), intermediate_dim=256, latent_dim=latent_dim)
    # conv
    elif AE_type == 'conv':
        model = Conv_Autoencoder(dim)

    elif AE_type == 'conv_deep':
        model = Conv_Deep_Autoencoder(dim)

    elif AE_type == 'conv_deep_v2':
        model = Conv_Deep_Autoencoder_v2(dim)

    elif AE_type == 'pca':
        model = PCA_Autoencoder(np.prod(dim), np.prod(dim), latent_dim)
    else:
        raise Exception('Model not yet implemented')

    model.load_state_dict(torch.load(path))
    model.eval()
    return model
