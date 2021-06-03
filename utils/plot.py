import matplotlib.cm as mpl_color_map
import cv2
import os
import torch
import torch.nn as nn
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE


def weight2color(weight_im):
    """
    colorize weight map

    Args:
        weight_im ([(0,1) array of BxHxW]): weight/mask maps
    """
    color_map = mpl_color_map.get_cmap('rainbow')
    heatmap = color_map(weight_im) * 255
    return heatmap


def tensor2rgb(tensor):
    tensor = tensor.numpy().squeeze().transpose(0, 2, 3, 1)

    # Scale between 0-255 to visualize

    tensor = np.uint8((tensor+1.)/2 * 255)
    # tensor = np.uint8(Image.fromarray(tensor)
    #                     .resize((256, 256), Image.ANTIALIAS))
    return tensor


def heatmaped_img(tensor_im, weight_im, to_tensor=True):
    img = tensor2rgb(tensor_im)
    output = []
    for i in range(img.shape[0]):
        w_im = weight_im[i]
        im = img[i]
        heatmap = weight2color(w_im)
        att = cv2.cvtColor(heatmap.astype(np.uint8), cv2.COLOR_RGBA2RGB)
        att = cv2.resize(att, (im.shape[0], img.shape[1]))
        dst = cv2.addWeighted(im.astype(np.float32), 0.7,
                              att.astype(np.float32), 0.3, 0)
        output.append(np.clip(dst.astype(np.uint8), 0, 255))

    output = np.array(output)

    if to_tensor:
        output = torch.from_numpy(output)
        output = output.permute(0, 3, 1, 2)
        output = output/255*2 - 1
        # print(output.size())

    return output


def plot_fc_weight(sample, weight):
    if sample.size(1) == 1:
        sample = sample.repeat(1, 3, 1, 1)  # gray 2 rgb

    plot_list = []
    plot_list.append(sample)

    B, C, H, W = sample.size()

    weight -= torch.min(weight)
    weight /= torch.max(weight)
    weight = weight.contiguous().view(H, W).unsqueeze(0)
    weight = weight.repeat(B, 1, 1).numpy()

    plot_list.append(heatmaped_img(sample, weight))
    plot_list = torch.cat(plot_list, dim=0)

    return plot_list


def plot_gram_cam(data, grad_cam):
    # plot_list = []
    # plot_list.append(data.cpu().detach())

    grad_cams = grad_cam.generate_cam(data)
    cvt = heatmaped_img(data.cpu().detach(), grad_cams)
    # plot_list.append(cvt)
    # plot_list = torch.cat(plot_list, dim=0)

    return cvt


def plot_embedding(X, y, model_tag):
    """
    Plot an embedding X with the class label y colored by the domain d.
    :param X: embedding
    :param y: label
    :return:
    """
    # normalization
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    N_class = 10
    for c in range(N_class):
        plt.scatter(X[y == c, 0], X[y == c, 1], color=plt.cm.rainbow(
            c/N_class), label='{}'.format(c+1))

    plt.xticks([]), plt.yticks([])

    # plt.legend(handles=[l1, l2], labels=[
    #            'target domain', 'source domain'], loc='best')
    plt.legend(loc='best')

    title = model_tag+" t-SNE"
    plt.title(title)

    model, layer = model_tag.split('+')
    dir_name = os.path.join('saved', 'imgs', 'tsne', model)
    os.makedirs(dir_name, exist_ok=True)

    imgName = os.path.join(dir_name, layer+'.png')

    print('Saving ' + imgName + ' ...')
    plt.savefig(imgName)
    plt.close()


def plot_tsne(embedings, labels, model='DANN', labels_gt=None):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    vecs = tsne.fit_transform(embedings)

    plot_embedding(vecs, labels, model_tag=model)
    if labels_gt is not None:
        plot_embedding(vecs, labels_gt, model_tag=model+"_GT")


def plot_lda(embedings, labels, model='DANN'):
    lda = LDA(n_component=2).to(labels.device)
    lda.fit(embedings, labels)
    vecs = lda.project(embedings)
    # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    # dann_tsne = tsne.fit_transform(embedings)

    plot_embedding(vecs.cpu().numpy(), labels.cpu().numpy(),
                   model_tag=model+'_lda')
