# PolSAM

**Polarimetric Scattering Mechanism Informed Segment Anything Model**

IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2026

**Our website: [https://wanda36.github.io/PolSAM/](https://wanda36.github.io/PolSAM/)**

[Journal paper](https://doi.org/10.1109/TCSVT.2026.3738868) · [arXiv](https://arxiv.org/abs/2412.12737)

## Getting started

### 1. Environment

Use a CUDA-enabled PyTorch installation compatible with your GPU and driver, together with the matching torchvision version. The repository does not currently provide a pinned dependency environment. Other packages imported by the training and evaluation code include:

```bash
python -m pip install numpy pandas Pillow opencv-python scipy scikit-learn \
  matplotlib seaborn imageio einops timm easydict tqdm icecream wandb
```

### 2. Prepare datasets and SAM weights

Extract the downloaded datasets into `datasets/`. The default manifest paths in `datasets/transforms.py` are:

```text
datasets/
├── PhySAR-Seg-1/
│   ├── PhySAR-Seg-1_train.csv
│   ├── PhySAR-Seg-1_val.csv
│   ├── PhySAR-Seg-1_test16.csv
│   └── ... image / MVD / label files
└── PhySAR-Seg-2/
    ├── PhySAR-Seg-2_train.csv
    ├── PhySAR-Seg-2_val.csv
    └── ... image / MVD / label files
```

Training CSVs must have a header row and three columns in this order: **pseudo-colored image path, MVD path, semantic label path**. Update paths inside the CSVs for your machine; relative paths are resolved from the working directory. See `dataset.py` for the separate test loader format and `datasets/transforms.py` for manifest paths and label palettes.

Download the [SAM pretrained weights](https://pan.baidu.com/s/15t1LyIlsn56aSr4euL8sQQ). The default ViT-B checkpoint path is:

```text
checkpoint/sam/sam_vit_b_01ec64.pth
```

### 3. Configure and train

Run commands from the repository root. Edit `config.py` before starting:

| Setting | Purpose |
| :--- | :--- |
| `dataset_name` | `coarse` for PhySAR-Seg-1; `SF` for PhySAR-Seg-2 |
| `ckpt`, `vit_name` | Matching SAM checkpoint and encoder |
| `device`, `n_gpu` | CUDA device index and GPU count; use `n_gpu = 1` for the workflow below |
| `batch_size`, `max_epochs`, `base_lr` | Training budget and optimization |
| `pt`, `adapter`, `Cnn_extra` | Model component switches |
| `train_group` | Encoder tuning scope; implementation in `Sam.set_trainable_group` |
| `loss` | `ce` for PhySAR-Seg-1; the paper uses focal loss for PhySAR-Seg-2 |
| `output`, `experiment` | Output directory and experiment name |

The current configuration contains local experiment settings (`experiment = 'debug'`, `train_group = 'E'`, `adapter = False`, CUDA device `3`). Review these settings before reproducing a paper experiment. Group A selects adapter/prompt parameters in the image encoder; B tunes the full image encoder; C/D/E select the last 1/2, 1/4, or 3/4 of ViT-B blocks, together with neck/prompt parameters. Other model components retain their configured trainability.

```bash
python train.py
```

Training writes `config.txt`, `evaluation.txt`, and the selected `result.pth` checkpoint to an experiment subdirectory under `save_model/` by default.

### 4. Evaluate

In `test.py`, set `netpath` to your trained `result.pth` and align the `torch.load(..., map_location=...)` device with your GPU. Match the model switches to the training configuration; the script currently contains a local absolute checkpoint path.

```bash
python test.py --dataset_name coarse --gpu_device 0
```

PhySAR-Seg-2 has no default test manifest (`path_test = None`); adapt the evaluation split and loader before evaluating it with `test.py`.

## Citation

If you use PolSAM or PhySAR-Seg, please cite:

```bibtex
@article{wang2026polsam,
  title   = {PolSAM: Polarimetric Scattering Mechanism Informed Segment Anything Model},
  author  = {Wang, Yuqing and Huang, Zhongling and Yang, Shuxin and Tang, Hao and Qiu, Xiaolan and Han, Junwei and Zhang, Dingwen},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2026},
  doi     = {10.1109/TCSVT.2026.3738868}
}
```

## Acknowledgments

Built on [Segment Anything](https://github.com/facebookresearch/segment-anything). See [LICENSE](LICENSE) for the repository license.
