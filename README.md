# DCDicL for Image Denoising
Hongyi Zheng*, Hongwei Yong*, Lei Zhang, "Deep Convolutional Dictionary Learning for Image Denoising," in CVPR 2021. (* Equal contribution)

[[paper]](https://www4.comp.polyu.edu.hk/~cslzhang/paper/DCDicL-cvpr21-final.pdf) [[supp]](https://www4.comp.polyu.edu.hk/~cslzhang/paper/DCDicL-cvpr21-supp.pdf)

The implementation of DCDicL is based on the awesome Image Restoration Toolbox [[KAIR]](https://github.com/cszn/KAIR).

## Requirement
- PyTorch 1.6+
- prettytable
- tqdm

## Testing
**Step 1**

- Download pretrained models from [[OneDrive]](https://1drv.ms/u/s!ApI9l49EgrUbjJw58CGV6lS4WCWpRw?e=3eJ7YF).
- Unzip downloaded file and put the folders into ```./release/denoising```

**Step 2**

Configure ```options/test_denoising.json```. Important settings:
- task: task name.
- path/root: path to save the tasks
- path/pretrained_netG: path to the folder containing the pretrained models.
- data/n_channels: 1 for greyscale and 3 for color.
- test/visualize: true for saving the noisy input/pkredicted dictionaries.

**Step 3**
```bash
python test_dcdicl.py
```



## Training
Training code will be released soon.
