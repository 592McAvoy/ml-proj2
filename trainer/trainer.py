import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils.gradcam import GradCam
from utils.ramps import sigmoid_rampup

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_valid = True #self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 50 #int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            'loss', 'kld_loss', 'recons_loss',
            # *[m.__name__ for m in self.metric_ftns],
            writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', 'kld_loss', 'recons_loss', 
            # *[m.__name__ for m in self.metric_ftns],
            writer=self.writer)

        # self.grad_cam = GradCam(self.model)

        self.fix_noise = None

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        # (1) from the Pytorch_VAE
        # kld_weight = batchsize / image_amount 
        # kld_weight = 1./len(self.data_loader)

        # (2) from the last answaer 
        # in https://stats.stackexchange.com/questions/332179/how-to-weight-kld-loss-vs-reconstruction-loss-in-variational-auto-encoder
        # kld_weight = latent_size / image_size(H*W)
        D = self.config['arch']['args']['latent_dim']
        H = self.config['train_loader']['args']['im_res']
        kld_norm = D/(H*H*3)
        w_kld = self.config['w_kld']
        # w_kld = sigmoid_rampup(current=epoch-1, rampup_length=self.epochs//4)
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output, mu, logvar = self.model(data)
            kld_loss = self.model.kld_loss(mu, logvar)
            recons_loss = self.model.recons_loss(data, output)
            loss = w_kld*kld_norm*kld_loss+recons_loss
            # loss = recons_loss

            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('kld_loss', kld_loss.item())
            self.train_metrics.update('recons_loss', recons_loss.item())

            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))

            if (batch_idx+1) % self.log_step == 0:
                logstr = 'Train Epoch: {} {} Loss: {:.6f} [KLD: {:.6f}, Recons: {:.6f}] W_KLD:{:.6f}'.format(
                    epoch, self._progress(batch_idx),
                    self.train_metrics.current('loss')/self.log_step,
                    self.train_metrics.current('kld_loss')/self.log_step,
                    self.train_metrics.current('recons_loss')/self.log_step,
                    w_kld)

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                # # log metrics
                # for met in self.metric_ftns:
                #     logstr = logstr + \
                #         " {}: {:.3f}".format(
                #             met.__name__, self.train_metrics.current(met.__name__)/self.log_step)

                self.train_metrics.log_all(log_step=self.log_step)
                self.logger.debug(logstr)

                # visulization
                N_sample = 8
                vis_im = torch.cat([
                    data[:N_sample, ...],
                    output[:N_sample, ...]
                ], dim=0)

                self.writer.add_image('recons',
                                      make_grid(vis_im.detach().cpu(),
                                                nrow=N_sample, normalize=True))

                # self.writer.add_image('input',
                #                       make_grid(plot_gram_cam(data[:64], self.grad_cam),
                #                                 nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_valid:
            val_log = self._valid_epoch(epoch)
            # log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about valid
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            if self.fix_noise is None:
                N = 64
                D = self.config['arch']['args']['latent_dim']
                self.fix_noise = torch.randn((N, D)).to(self.device)

            gen_sample = self.model.decode(self.fix_noise)

            # for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            #     data, target = data.to(self.device), target.to(self.device)

            #     output = self.model(data)
            #     loss = self.criterion(output, target)

            #     # update record
            #     self.valid_metrics.update('loss', loss.item())
            #     for met in self.metric_ftns:
            #         self.valid_metrics.update(
            #             met.__name__, met(output, target))
            #     # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        self.writer.set_step(
            epoch - 1, 'valid')
        # self.valid_metrics.log_all()
        self.writer.add_image('generate',
                              make_grid(gen_sample.detach().cpu(),
                                        nrow=8, normalize=True))
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
