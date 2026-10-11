import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import warnings

from scipy.ndimage import gaussian_filter

from tools.evaluation_metrics import Xauc, perimeter, area
from matplotlib.ticker import FormatStrFormatter


def weighted_htmaps(htmaps, n_pixels=1):
    w_htmaps = np.empty_like(htmaps)
    for k in range(len(htmaps)):
        ht = htmaps[k]
        for i in range(ht.shape[0]):
            for j in range(ht.shape[1]):
                idx_i = np.arange(i - n_pixels, i + n_pixels + 1)
                idx_i = idx_i[np.where((idx_i >= 0) & (idx_i < ht.shape[-2]))[0]]
                idx_j = np.arange(j - n_pixels, j + n_pixels + 1)
                idx_j = idx_j[np.where((idx_j >= 0) & (idx_j < ht.shape[-2]))[0]]
                s = ht[idx_i][:, idx_j].sum() - ht[i, j]
                w_htmaps[k, i, j] = ht[i, j] * (s / (len(idx_i) * len(idx_j) - 1))
    return w_htmaps


def plot_image(image, cmap='gray'):
    '''
    Method to plot an image
    :param image: (C x H x W) channel-first-format image
    :return:
    '''
    if image.shape[0] == 1:
        image = image[0, :, :]
    else:
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 1, 2)

    plt.imshow(image, cmap=cmap)
    plt.show()


def plot_heatmap(image, colorbar=True):
    '''
    Method to plot a heatmap
    :param image: (1 x H x W) channel-first-format image
    :return:
    '''
    image = image[0, :, :]

    plt.imshow(image, cmap='jet')#, vmin=0.0, vmax=1.0)
    if colorbar:
        plt.colorbar()
    plt.show()


def plot_results(X_test, Y_test, GT_test, htmaps, ids):
    plt.figure(figsize=(9,15))
    for i in range(5):
        plt.subplot(5, 3, (i * 3) + 1)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_test[ids[i]])
        plt.subplot(5, 3, (i * 3) + 2)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(htmaps[ids[i]], cmap='jet', vmin=htmaps[ids[i]].min(), vmax=htmaps[ids[i]].max())
        plt.xlabel(f'AUC: {round(Xauc(GT_test[ids[i]], htmaps[ids[i]]), 4)}')
        #plt.colorbar()
        plt.subplot(5, 3, (i * 3) + 3)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(GT_test[ids[i]], cmap='jet')
    plt.show()



def rad_from_lines(gt, d_thres=3):
    nc = gt.shape[1]
    gtrow = gt.cumsum(axis=1) * gt
    drow = gtrow[np.where(gtrow[:, 0:nc - 1] > gtrow[:, 1:nc])]
    drow = np.concatenate((drow, gtrow[gtrow[:, nc - 1] > 0, nc - 1]))
    drow = drow[drow > d_thres]
    if drow.shape[0] == 0:
        rrow = 0
    else:
        rrow = 0.5 * drow.mean()
    nr = gt.shape[0]
    gtcol = gt.cumsum(axis=0) * gt
    dcol = gtcol[np.where(gtcol[0:nr - 1, :] > gtcol[1:nr, :])]
    dcol = np.concatenate((dcol, gtcol[nr - 1, gtcol[nr - 1, :] > 0]))
    dcol = dcol[dcol > d_thres]
    if dcol.shape[0] == 0:
        rcol = 0
    else:
        rcol = 0.5 * dcol.mean()
    return max(rcol, rrow)


def score_neighs(ht, rad=1):
    sc = 0
    for i in range(ht.shape[0]):
        for j in range(ht.shape[1]):
            l = max(0, j - rad)
            r = min(ht.shape[1], j + rad + 1)
            u = max(0, i - rad)
            d = min(ht.shape[0], i + rad + 1)
            w = np.sum(ht[u:d, l:r]) - ht[i, j]
            w = w /((r-l)*(d-u)-1)
            sc = sc + ht[i, j]*w
    return sc


def scores_neighs(hts, rad=1):
    scores = np.zeros(hts.shape[0])
    for i in range(hts.shape[0]):
        ht = hts[i]
        scores[i] = score_neighs(ht, rad)
    return scores

def get_filter_sigma(ht, thres=0.5, scale=0.25):
    if ht.max() > 1:
        warnings.warn('Error: heatmap not in [0,1]')
    binht = np.where(ht >= thres, 1, 0)
    rad = rad_from_lines(binht)
    return max(rad * scale, 2)

def blurred_htmaps_auto(hts, thres=0.5, scale=0.25, return_sigma=False):
    hts_b = np.empty_like(hts)
    sigmas = np.empty(hts.shape[0])
    for i in range(hts.shape[0]):
        sigma = get_filter_sigma(hts[i], thres=thres, scale=scale)
        sigmas[i] = sigma
        hts_b[i] = gaussian_filter(hts[i], sigma=sigma)

    if return_sigma:
        return hts_b, sigmas
    return hts_b

def blurred_htmaps(hts, k):
    hts_b = np.empty_like(hts)
    for i in range(hts.shape[0]):
        hts_b[i] = gaussian_filter(hts[i], sigma=k/3)
    return hts_b


def plot_results_radii(mean_res, values, class_l, y_label, postfix='radii_t', metric="$k$"):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    plt.title('Class ' + class_l)
    plt.ylabel(metric)
    plt.xlabel(y_label)
    #plt.ylim(0, 8)
    ax.plot(values, mean_res, 'ro-')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.grid()
    plt.savefig(f'{class_l}_{postfix}.eps')
    plt.close()


def plot_aucs_at(aucs_k, aucs_t, radii, k_values, ats, class_l):
    plt.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    plt.title('Class ' + class_l)
    plt.ylabel('AUC')
    plt.xlabel("$k$")
    plt.ylim(0.75, 1.001)
    unique_ats = np.unique(ats)
    for i in range(len(unique_ats)):
        at = unique_ats[i]
        auc_m = aucs_k[:, ats == at].mean(axis=-1)
        auc_s = aucs_k[:, ats == at].std(axis=-1)
        ax.plot(k_values, auc_m, '-', color=f'C{i}', label=at)
        #ax.plot(k_values, np.where(auc_m + auc_s > 1, 1,
        #                         auc_m + auc_s), '--', color=f'C{i}')
        #ax.plot(k_values, np.where(auc_m - auc_s < 0, 0,
        #                         auc_m - auc_s), '--', color=f'C{i}')
        ax.scatter(int(radii[ats == at].mean()), aucs_t[ats == at].mean(), marker='x', color=f'C{i}')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.grid()
    plt.legend()
    #plt.show()
    plt.savefig(f'{class_l}_k.eps')
    plt.close()


def plot_filter_results_mvtec(results, results_std, labels, kws, save=False, file_name=None, ids=None):
    fig, axs = plt.subplots(3, 5, sharex=True, sharey='row', figsize=(27, 10))
    fig.tight_layout(w_pad=2, h_pad=2, rect=(0.08, 0.102, 0.992, 0.945))
    fig.supylabel('AUC', fontsize=24)
    fig.supxlabel("$k$", fontsize=24)
    #plt.ylim(0.75, 1.001)
    plt.xticks(range(0, 51, 10))
    #plt.yticks(np.arange(0., 1.001, 0.2))
    if ids is None:
        ids = list(range(15))
    for i in range(15):
        i_id = ids[i]
        axs[i // 5][i % 5].set_title(labels[i_id], fontsize=24)
        axs[i // 5][i % 5].plot(kws, results[i_id], 'r-')
        axs[i // 5][i % 5].plot(kws, np.where(results[i_id] + results_std[i_id] > 1, 1,
                                              results[i_id] + results_std[i_id]), 'r--')
        axs[i // 5][i % 5].plot(kws, np.where(results[i] - results_std[i] < 0, 0,
                                              results[i_id] - results_std[i_id]), 'r--')
        axs[i // 5][i % 5].grid(True)
        axs[i // 5][i % 5].tick_params(labelsize=24)
        axs[i // 5][i % 5].tick_params('x', rotation=45)
        axs[i // 5][i % 5].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if save:
        name = 'k_results_mvtec.jpg' if file_name is None else file_name
        fig.savefig(name)
    plt.show()

