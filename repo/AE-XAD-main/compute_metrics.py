import os

from sklearn.metrics import roc_auc_score
from tools.evaluation_metrics import Xauc
import numpy as np

from utils import blurred_htmaps
from tools.load_data import load_model
from torchvision import transforms
from PIL import Image

if __name__ == '__main__':
    dataset = 'btad_01'
    na = 10
    root = f'results/{dataset}/{na}'
    model = 'conv_deep_v2'

    gts = np.load(os.path.join(root, 'aexad_gtmaps_norm.npy'))[:, 0]
    y = np.load(os.path.join(root, 'aexad_labels_norm.npy'))
    ht_aexad = np.load(os.path.join(root, f'aexad_htmaps_{model}.npy'))
    sc_aexad = np.load(os.path.join(root, f'aexad_scores_{model}.npy'))
    ht_aexad = ht_aexad.mean(axis=1)

    ht_aexad_blur = np.empty((ht_aexad.shape[0], ht_aexad.shape[1], ht_aexad.shape[2], 3))
    scales = [0.5, 1., 1.5]
    for i in range(len(scales)):
        scale = scales[i]
        ht_aexad_blur[:, :, :, i] = blurred_htmaps(ht_aexad, scale=scale)

    nat = int(y.sum())
    ht_auc_aexad = np.empty(nat)
    ht_auc_aexad_blur = np.empty((nat, 3))
    for i in range(nat):
        if gts[y==1][i].sum() == 0:
            ht_auc_aexad[i] = -1
        else:
            ht_auc_aexad[i] = Xauc(gts[y==1][i], ht_aexad[y==1][i])
            ht_auc_aexad_blur[i, 0] = Xauc(gts[y==1][i], ht_aexad_blur[y==1][i, :, :, 0])
            ht_auc_aexad_blur[i, 1] = Xauc(gts[y==1][i], ht_aexad_blur[y==1][i, :, :, 1])
            ht_auc_aexad_blur[i, 2] = Xauc(gts[y==1][i], ht_aexad_blur[y==1][i, :, :, 2])

    print('EXP AUC:', ht_auc_aexad.mean())
    print('EXP AUC scales:', ht_auc_aexad_blur.mean(axis=0))
    print('DET AUC: ', roc_auc_score(y, sc_aexad))

    model = load_model(os.path.join(root, f'aexad_model_{model}.pt'), (3, 448, 448), AE_type=model, latent_dim=64)

    X_test = np.load(open(os.path.join(f'data/{dataset}', 'X_test.npy'), 'rb'))
    Y_test = np.load(open(os.path.join(f'data/{dataset}', 'Y_test.npy'), 'rb'))
    GT_test = np.load(open(os.path.join(f'data/{dataset}', 'GT_test.npy'), 'rb'))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448), Image.NEAREST),
        # transforms.ToTensor(),
        transforms.PILToTensor()
    ])

    X_test_res = np.empty((X_test.shape[0], 3, 448, 448))
    for i in range(X_test.shape[0]):
        X_test_res[i] = transform(X_test[i])

    X_test_res = (X_test_res / 255.).astype(np.float32)
