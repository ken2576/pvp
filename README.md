# PVP: Personalized Video Prior for Editable Dynamic Portraits using StyleGAN


## Requirements

Install dependencies.

`conda create -n pvp python=3.9`

`pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`

`conda install hydra-core hydra-colorlog hydra-optuna-sweeper tqdm pytorch-lightning==1.8.1 torchmetrics kornia==0.6.7 scipy scikit-image`

`conda install bottler nvidiacub`

`pip install timm tensorboardX blobfile gpustat torchinfo fairseq==0.10.0 click einops safetensors chumpy face_alignment`

`pip install git+https://github.com/facebookresearch/pytorch3d.git`

`conda activate pvp`

Download the following and put them in `./assets`

| Path | Description |
|---| ---|
| [FFHQ StyleGAN](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl)  | StyleGAN2-ada model trained on FFHQ with 1024x1024 output resolution. |
| [ArcFace](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | ArcFace model for identity loss. |



## Data Preprocessing

### Run NHA preprocessing to generate poses

See [NHA](https://github.com/philgras/neural-head-avatars#optimizing-an-avatar-against-a-monocular-rgb-video)

### Run masking to generate masked data

```python make_masked_images.py --folder [nha processed data]```

### Extract FLAME data for each frame with DECA

```python extract_expr_cli.py --folder [nha processed data]```

### Generate key frames for PTI

```python make_keyframes_expr.py --folder [nha processed data] --out_folder [keyframe folder] --n_cluster 200 --n_sample 1 --val_start [validation frame start]```

### Run PTI on the data

```python -m scripts.run_pti --input_data_path [keyframe folder] --input_data_id [name] --experiments_output_dir [PTI output folder]```

## Training

Set up the configs (`train.yaml`), and replace paths starting with `/home/ken` with your own paths.

Change `train_size` and `test_size` in `train.yaml` to your desired sizes.
Default `train_size=750` means the first 750 frames will be used as training data. `test_size=700` means frame 751 to 1450 will be used as test data.

Run `python train.py --config-name=train name="[your experiment name]"`
