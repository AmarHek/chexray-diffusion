# Cascaded Latent Diffusion for High-Resolution Chest X-ray Synthesis

<p align="center">
<img src=assets/intro_sample_grid.png />
</p>

This repository contains code for running and training **Cheff** - a cascaded **che**st
X-ray latent di**ff**usion pipeline.
The cheff pipeline consists of three cascading phases:

1. Modeling a diffusion process in latent space
2. Translating the latent variables into image space with a decoder
3. Refinement and upscaling using a super-resolution diffusion process

Phase 1 and 2 together define an LDM.
Phase 2 and 3 are trained on MaCheX, a collection of over 650,000 chest X-rays and thus,
build a foundational basis for our model stack.
The first phase is task-specific. For unconditional snythesis, we train on full MaCheX
and for report-to-chest-X-ray we use the MIMIC subset.

<p align="center">
<img src=assets/cheff_overview.png />
</p>

## How to use Cheff?

Please have a look into our [tutorial notebook](notebooks/01_cheff.ipynb).


## Models

We provide the weights for 5 models:

- Chest X-ray autoencoder: [Click!](https://syncandshare.lrz.de/getlink/fiQ6wTe7K7otQzyifNh9av/cheff_autoencoder.pt)
- Chest X-ray unconditioned semantic diffusion model: [Click!](https://syncandshare.lrz.de/getlink/fiE9pKbK38wzEvBrBCk95W/cheff_diff_uncond.pt)
- Chest X-ray report-conditioned semantic diffusion model: [Click!](https://syncandshare.lrz.de/getlink/fi4R87B3cEWgSx4Wivyizb/cheff_diff_t2i.pt)
- Chest X-ray super-resolution diffusion model base: [Click!](https://syncandshare.lrz.de/getlink/fiovQdSGXiTuWQ7scu7FA/cheff_sr_base.pt)
- Chest X-ray super-resolution diffusion model finetuned: [Click!](https://syncandshare.lrz.de/getlink/fiHM4uAfy7uxcfBXkefySJ/cheff_sr_fine.pt)

The [tutorial notebook](notebooks/01_cheff.ipynb) assumes that downloaded models are
placed in `trained_models`.

## Training

Our codebase builds heavily on the classic LDM repository. Thus, we share the same
interface with a few adaptions.
A conda environment file for installing necessary dependencies is `environment.yml`.
For a pip-only install use `requirements.txt`.
The full config files are located in `configs`. After adjusting the paths, the training
can be started as follows:

```shell
python scripts/train_ldm.py -b <path/to/config.yml> -t --no-test
```

### Training the Super-Resolution Model

The training procedure for reproducing `CheffSR` is located in an [extra repository](https://github.com/saiboxx/diffusion-pytorch).
You will find a [script](https://github.com/saiboxx/diffusion-pytorch/blob/main/scripts/03_train_sr3_ddp.py) that contains the 
necessary configuration and routine.


## Acknowledgements

This code builds heavily on the implementation of LDMs and DDPMs from CompVis:
[Repository here](https://github.com/CompVis/latent-diffusion).


## Citation

If you use **Cheff** or our repository in your research, please cite our paper *Cascaded Latent Diffusion Models for High-Resolution Chest X-ray Synthesis*:

```
@inproceedings{weber2023cascaded,
  title={Cascaded Latent Diffusion Models for High-Resolution Chest X-ray Synthesis},
  author={Weber, Tobias and Ingrisch, Michael and Bischl, Bernd and R{\"u}gamer, David},
  booktitle={Advances in Knowledge Discovery and Data Mining: 27th Pacific-Asia Conference, PAKDD 2023},
  year={2023},
  organization={Springer}
}
```
