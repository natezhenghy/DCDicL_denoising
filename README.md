# Deep Convolutional Dictionary Learning for Image Denoising
Hongyi Zheng*, Hongwei Yong*, Lei Zhang, "Deep Convolutional Dictionary Learning for Image Denoising," in CVPR 2021. (* Equal contribution)

[[paper]](https://www4.comp.polyu.edu.hk/~cslzhang/paper/DCDicL-cvpr21-final.pdf) [[supp]](https://www4.comp.polyu.edu.hk/~cslzhang/paper/DCDicL-cvpr21-supp.pdf)

The implementation of DCDicL is based on the awesome Image Restoration Toolbox [[KAIR]](https://github.com/cszn/KAIR).

## Requirement
- PyTorch 1.6+
- prettytable
- tqdm

## Testing
**Step 1**

- Download pretrained models from [[OneDrive]](https://1drv.ms/u/s!ApI9l49EgrUbnK5zR1R2F8ZP8z1NjQ?e=L3MmSf) or [[BaiduPan]](https://pan.baidu.com/share/init?surl=vIqN2XiZ9UH8vcUpZPbXnw) (password: flfw).
- Unzip downloaded file and put the folders into ```./release/denoising```

**Step 2**

Configure ```options/test_denoising.json```. Important settings:
- task: task name.
- path/root: path to save the tasks.
- path/pretrained_netG: path to the folder containing the pretrained models.
- data/n_channels: 1 for greyscale and 3 for color.
- test/visualize: true for saving the noisy input/predicted dictionaries.

**Step 3**
```bash
python test_dcdicl.py
```



## Training

If you want to achieve the best performance:
- you have to first train a 1-stage model, then train a multi-stage (2~6) model based on the pretrained model. (Please refer the paper for more details.)
- you have to include [[Waterloo Exploration Database]](https://ece.uwaterloo.ca/~k29ma/exploration/) in the training sets.

**Step 1**

Prepare training/testing data. The folder structure should be similar to:

```
+-- data
|   +-- train
|       +-- training_dataset_1
|       +-- training_dataset_2
|   +-- test
|       +-- testing_dataset_1
|       +-- testing_dataset_2
```

**Step 2**

Configure ```options/train_denoising.json```. Important settings:
- task: task name.
- path/root: path to save the tasks.
- data/n_channels: 1 for greyscale and 3 for color.
- data/train/sigma: range of noise levels.
- netG/d_size: dictionary size.
- netG/n_iter: number of iterations.
- netG/nc_x: number of channels in NetX.
- netG/nb: number of blocks in NetX.
- test/visualize: true for saving the noisy input/predicted dictionaries.

If you want to reload a pretrained model, pay attention to following settings:
- path/pretrained_netG: path to the folder containing the pretrained models.
- train/reload_broadcast: if you want to load a pretrained 1-stage model into multi-stage model, please set this item to **true**.


**Step 3**
```bash
python train_dcdicl.py
```

**FAQ**
- Keep receiving ''WARNING batched routines are designed for mall sizes. It might be ...''.

This is the limitation of the backend linear algebra GPU accelerated libraries of PyTorch. The only way to get rid of it is to reduce the number of channels or spatial size of the dictionaries.

## Citation
```
@InProceedings{Zheng_2021_CVPR,
    author    = {Zheng, Hongyi and Yong, Hongwei and Zhang, Lei},
    title     = {Deep Convolutional Dictionary Learning for Image Denoising},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {630-641}
}
```
