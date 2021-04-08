import argparse
from data.dataset_denoising import DatasetDenoising
import logging
import os
import os.path
from typing import Dict, List

from prettytable import PrettyTable
from torch.utils.data import DataLoader

from data.dataset_ir import DatasetIR
from data.select_dataset import select_dataset
from models.model import Model
from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option


def main(config_path: str = 'options/test_denoising.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',
                        type=str,
                        default=config_path,
                        help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.makedirs(
        [path for key, path in opt['path'].items() if 'pretrained' not in key])

    option.save(opt)

    logger_name = 'test'
    utils_logger.logger_info(
        logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    opt_data_test = opt["data"]["test"]
    test_sets: List[DatasetDenoising] = select_dataset(opt_data_test, "test")
    test_loaders: List[DataLoader[DatasetDenoising]] = []
    for test_set in test_sets:
        test_loaders.append(
            DataLoader(test_set,
                       batch_size=1,
                       shuffle=False,
                       num_workers=1,
                       drop_last=False,
                       pin_memory=True))

    model = Model(opt)
    model.init()

    avg_psnrs: Dict[str, List[float]] = {}
    avg_ssims: Dict[str, List[float]] = {}
    tags = []
    for test_loader in test_loaders:
        test_set: DatasetDenoising = test_loader.dataset
        avg_psnr = 0.
        avg_ssim = 0.
        for test_data in test_loader:
            model.feed_data(test_data)
            model.test()

            psnr, ssim = model.cal_metrics()
            avg_psnr += psnr
            avg_ssim += ssim

            model.save_visuals(test_set.tag)

        avg_psnr = round(avg_psnr / len(test_loader), 2)
        avg_ssim = round(avg_ssim * 100 / len(test_loader), 2)

        name = test_set.name

        if name in avg_psnrs:
            avg_psnrs[name].append(avg_psnr)
            avg_ssims[name].append(avg_ssim)
        else:
            avg_psnrs[name] = [avg_psnr]
            avg_ssims[name] = [avg_ssim]

        tags.append(test_set.tag)

    header = ['Dataset'] + list(set(tags))

    t = PrettyTable(header)
    for key, value in avg_psnrs.items():
        t.add_row([key] + value)
    logger.info(f"Test PSNR:\n{t}")

    t = PrettyTable(header)
    for key, value in avg_ssims.items():
        t.add_row([key] + value)
    logger.info(f"Test SSIM:\n{t}")


main()
