import copy
from logging import Logger

from models.network_denoising import DCDicL
import os
from glob import glob
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.optim import Adam, lr_scheduler

from models.select_network import select_network
from utils import utils_image as util


class Model:
    def __init__(self, opt: Dict[str, Any]):
        self.opt = opt
        self.opt_train = self.opt['train']
        self.opt_test = self.opt['test']

        self.save_dir: str = opt['path']['models']
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.type = opt['netG']['type']

        self.net = select_network(opt).to(self.device)
        self.net = DataParallel(self.net)

        self.schedulers = []
        self.log_dict = {}
        self.metrics = {}

    def init(self):

        self.load()

        self.net.train()

        self.define_loss()
        self.define_optimizer()
        self.define_scheduler()

    def load(self):
        load_path = self.opt['path']['pretrained_netG']
        if load_path is not None:
            print('Loading model for G [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.net)

    def load_network(self, load_path: str, network: Union[nn.DataParallel,
                                                          DCDicL]):
        if isinstance(network, nn.DataParallel):
            network: DCDicL = network.module

        network.head.load_state_dict(torch.load(load_path + 'head.pth'),
                                     strict=True)

        state_dict_x = torch.load(load_path + 'x.pth')
        network.body.net_x.load_state_dict(state_dict_x, strict=True)

        state_dict_d = torch.load(load_path + 'd.pth')
        network.body.net_d.load_state_dict(state_dict_d, strict=True)

        state_dict_hypa = torch.load(load_path + 'hypa.pth')
        if self.opt['train']['reload_broadcast']:
            state_dict_hypa_v2 = copy.deepcopy(state_dict_hypa)
            for key in state_dict_hypa:
                state_dict_hypa_v2[key.replace(
                    '0.mlp', 'mlp')] = state_dict_hypa_v2.pop(key)
            for hypa in network.hypa_list:
                hypa.load_state_dict(state_dict_hypa, strict=True)
        else:
            network.hypa_list.load_state_dict(state_dict_hypa, strict=True)

    def save(self, logger: Logger):
        logger.info('Saving the model.')
        net = self.net
        if isinstance(net, nn.DataParallel):
            net = net.module
        self.save_network(net.body.net_x, 'x')
        self.save_network(net.hypa_list, 'hypa')
        self.save_network(net.head, 'head')
        self.save_network(net.body.net_d, 'd')

    def save_network(self, network, network_label):
        filename = '{}.pth'.format(network_label)
        save_path = os.path.join(self.save_dir, filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)

    def define_loss(self):
        self.lossfn = nn.L1Loss().to(self.device)

    def define_optimizer(self):
        optim_params = []
        for _, v in self.net.named_parameters():
            optim_params.append(v)
        self.optimizer = Adam(optim_params,
                              lr=self.opt_train['G_optimizer_lr'],
                              weight_decay=0)

    def define_scheduler(self):
        self.schedulers.append(
            lr_scheduler.MultiStepLR(self.optimizer,
                                     self.opt_train['G_scheduler_milestones'],
                                     self.opt_train['G_scheduler_gamma']))

    def update_learning_rate(self, n: int):
        for scheduler in self.schedulers:
            scheduler.step(n)

    @property
    def learning_rate(self) -> float:
        return self.schedulers[0].get_lr()[0]

    def feed_data(self, data: Dict[str, Any]):
        self.y = data['y'].to(self.device)
        self.y_gt = data['y_gt'].to(self.device)

        self.sigma = data['sigma'].to(self.device)
        self.path = data['path']

    def cal_multi_loss(self, preds: List[torch.Tensor],
                       gt: torch.Tensor) -> torch.Tensor:
        losses = None
        for i, pred in enumerate(preds):
            loss = self.lossfn(pred, gt)
            if i != len(preds) - 1:
                loss *= (1 / (len(preds) - 1))
            if i == 0:
                losses = loss
            else:
                losses += loss
        return losses

    def log_train(self, current_step: int, epoch: int, logger: Logger):
        message = f'Training epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{self.learning_rate:.3e}'
        for k, v in self.log_dict.items(
        ):  # merge log information into message
            message += f', {k:s}: {v:.3e}'
        logger.info(message)

    def test(self):
        self.net.eval()

        with torch.no_grad():
            y = self.y
            h, w = y.size()[-2:]
            top = slice(0, h // 8 * 8)
            left = slice(0, (w // 8 * 8))
            y = y[..., top, left]

            self.dx, self.d = self.net(y, self.sigma)

        self.prepare_visuals()

        self.net.train()

    def prepare_visuals(self):
        """ prepare visual for first sample in batch """
        self.out_dict = {}
        self.out_dict['y'] = util.tensor2uint(self.y[0].detach().float().cpu())
        self.out_dict['dx'] = util.tensor2uint(
            self.dx[0].detach().float().cpu())
        self.out_dict['d'] = self.d[0].detach().float().cpu()
        self.out_dict['y_gt'] = util.tensor2uint(
            self.y_gt[0].detach().float().cpu())
        self.out_dict['path'] = self.path[0]

    def cal_metrics(self):
        self.metrics['psnr'] = util.calculate_psnr(self.out_dict['dx'],
                                                   self.out_dict['y_gt'])
        self.metrics['ssim'] = util.calculate_ssim(self.out_dict['dx'],
                                                   self.out_dict['y_gt'])

        return self.metrics['psnr'], self.metrics['ssim']

    def save_visuals(self, tag: str):
        y_img = self.out_dict['y']
        d_img = self.out_dict['d']
        dx_img = self.out_dict['dx']
        path = self.out_dict['path']

        img_name = os.path.splitext(os.path.basename(path))[0]
        img_dir = os.path.join(self.opt['path']['images'], img_name)
        os.makedirs(img_dir, exist_ok=True)

        old_img_path = os.path.join(img_dir, f"{img_name:s}_{tag}_*_*.png")
        old_img = glob(old_img_path)
        for img in old_img:
            os.remove(img)

        img_path = os.path.join(
            img_dir,
            f"{img_name}_{tag}_{self.metrics['psnr']}_{self.metrics['ssim']}.png"
        )

        util.imsave(dx_img, img_path)

        if self.opt['test']['visualize']:
            util.save_d(
                d_img.mean(0).numpy(), img_path.replace('.png', '_d.png'))
            util.imsave(y_img, img_path.replace('.png', '_y.png'))

    def train(self):
        self.optimizer.zero_grad()
        dxs, self.d = self.net(self.y, self.sigma)

        # loss
        loss = self.cal_multi_loss(dxs, self.y_gt)
        self.log_dict['G_loss'] = loss.item()

        self.dx = dxs[-1]
        loss.backward()

        self.optimizer.step()
