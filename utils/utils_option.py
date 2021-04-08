import json
import os
from datetime import datetime
from typing import Any, Dict

import commentjson
'''
# --------------------------------------------
# Hongyi Zheng (github: https://github.com/natezhenghy)
# 07/Apr/2021
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------

'''


def get_timestamp() -> str:
    return datetime.now().strftime('_%y%m%d_%H%M%S')


def parse(opt_path: str, is_train: bool = True) -> Dict[str, Any]:

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    with open(opt_path) as file:
        opt: Dict[str, Any] = commentjson.load(file)

    opt['opt_path'] = opt_path
    opt['is_train'] = is_train

    # ----------------------------------------
    # data
    # ----------------------------------------
    if 'scale' not in opt['data']:
        opt['data']['scale'] = 1

    # ----------------------------------------
    # datasets
    # ----------------------------------------
    for phase in ['train', 'test']:
        dataset = opt['data'][phase]
        dataset['type'] = opt['data']['type']
        dataset['phase'] = phase
        dataset['scale'] = opt['data']['scale']  # broadcast
        dataset['n_channels'] = opt['data']['n_channels']  # broadcast
        if 'k_size' in opt['data']:
            dataset['k_size'] = opt['data']['k_size']  # broadcast
        if 'dataroot_H' in dataset and dataset['dataroot_H'] is not None:
            dataset['dataroot_H'] = os.path.expanduser(dataset['dataroot_H'])
        if 'dataroot_L' in dataset and dataset['dataroot_L'] is not None:
            dataset['dataroot_L'] = os.path.expanduser(dataset['dataroot_L'])

    # ----------------------------------------
    # path
    # ----------------------------------------
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)

    path_task = os.path.join(opt['path']['root'], opt['task'])
    opt['path']['task'] = path_task
    opt['path']['log'] = path_task
    opt['path']['options'] = os.path.join(path_task, 'options')

    if is_train:
        opt['path']['models'] = os.path.join(path_task, 'models')
        opt['path']['images'] = os.path.join(path_task, 'images')
    else:  # test
        opt['path']['images'] = os.path.join(path_task, 'test_images')

    # ----------------------------------------
    # network
    # ----------------------------------------
    opt['netG']['type'] = opt['data']['type']
    opt['netG']['in_nc'] = opt['netG']['out_nc'] = opt['data']['n_channels']
    opt['netG']['scale'] = opt['data']['scale']

    # ----------------------------------------
    # GPU devices
    # ----------------------------------------
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return opt


'''
# --------------------------------------------
# convert the opt into json file
# --------------------------------------------
'''


def save(opt: Dict[str, Any]):
    opt_path = opt['opt_path']
    opt_path_copy = opt['path']['options']
    _, filename_ext = os.path.split(opt_path)
    filename, ext = os.path.splitext(filename_ext)
    dump_path = os.path.join(opt_path_copy, filename + get_timestamp() + ext)
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


'''
# --------------------------------------------
# dict to string for logger
# --------------------------------------------
'''


def dict2str(opt: Dict[str, Any], indent_l: int = 1):
    msg: str = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg
