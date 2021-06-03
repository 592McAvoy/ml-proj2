import cv2
import argparse
import torch
from tqdm import tqdm
import data_loader as module_data
import utils.loss as module_loss
import utils.metric as module_metric
# import model.model as module_arch
import model as module_arch
from parse_config import ConfigParser
from utils.gradcam import GradCam
from torchvision.utils import make_grid
# from utils.plot import plot_tsne, plot_gram_cam, plot_lda
import numpy as np
import os

def save_grid(im, spec, im_name):
    im = make_grid(im, nrow=im.size(0)//4, normalize=True)
    npimg = im.cpu().numpy().transpose(1, 2, 0)*255
    npimg = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
    # print(npimg.shape)
    im_path = os.path.join('saved/imgs/', spec)
    os.makedirs(im_path, exist_ok=True)
    cv2.imwrite(os.path.join(im_path,im_name), npimg)


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['train_loader']['type'])(
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # build model architecture, then print to console
    # if 'Kernel' in config["name"]:
    #     model = config.init_obj('arch', module_arch,
    #                             N=old_bs)
    # else:
    model = config.init_obj('arch', module_arch)
    # logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # grad_cam = GradCam(model)

    # total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        # image reconstrution
        # for data, target in data_loader:
        #     data, target = data.to(device), target.to(device)
        #     output, mu, logvar = model(data)

        #     mask = torch.ones((64, 64)).to(device)
        #     w = 20
        #     mask[32-w:32+w, 32-w:32+w] = 0.
        #     corrop_data = data * mask
        #     corrop_output, *_ = model(corrop_data)
        #     # print(corrop_data.size(), corrop_output.size())
        #     recons_sample = torch.cat([
        #         data, output, corrop_data, corrop_output
        #     ], dim=0)
        #     # print(recons_sample.size())
        #     save_grid(recons_sample, config["spec"], 'recons_sample_{}.png'.format(w))
        #     break
        # exit()

        # random noise generation
        # N = 64
        # D = config['arch']['args']['latent_dim']
        # fix_noise = torch.randn((N, D)).to(device)
        # gen_sample = model.decode(fix_noise)
        # save_grid(gen_sample, config["spec"], 'gen_sample.png')

        # # interpolation
        # h1, h2, h3, h4 = fix_noise[0:1, ...], fix_noise[15:16, ...], fix_noise[31:32, ...], fix_noise[47:48, ...]
        # tmp1 = []
        # tmp2 = []
        # for i in range(8):
        #     p = i/7
        #     tmp1.append(p*h1+(1-p)*h2)
        #     tmp2.append(p*h3+(1-p)*h4)
        # tmp1 = torch.stack(tmp1, dim=0)
        # tmp2 = torch.stack(tmp2, dim=0)

        # tmp = []
        # for i in range(8):
        #     p = i/7
        #     tmp.append(p*tmp1+(1-p)*tmp2)

        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            break
        mu, log_var = model.encode(data)
        z = model.reparameterize(mu, log_var)

        h1, h2 = z[:4, ...], z[4:, ...]
        tmp = []
        for i in range(8):
            p = i/7
            tmp.append(p*h1+(1-p)*h2)
        tmp = torch.cat(tmp, dim=0)   
        interp_noise = tmp     
        interp_sample = model.decode(interp_noise)
        im = torch.cat([data[4:, ...], interp_sample, data[:4, ...]], dim=0)
        print(im.size())
        N, C, H, W = im.size()
        im = im.contiguous().view(10,4,C, H, W).permute(1,0,2,3,4)
        print(im.size())
        im = im.contiguous().view(N,C,H,W)
        save_grid(im, config["spec"], 'interp_sample.png')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
