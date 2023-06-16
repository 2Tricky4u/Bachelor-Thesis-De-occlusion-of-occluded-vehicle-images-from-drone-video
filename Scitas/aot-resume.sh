#!/bin/bash
#SBATCH --chdir /home/xogay
#SBATCH --partition=gpu
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --time 00:05:00
#SBATCH --requeue

echo STARTING AT $(date)
echo "Job run at: $(hostname)"

module load gcc/9.3.0-cuda
module load python/3.7.7

source ~/.venvs/guided-diffusion-repaint/bin/activate
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate inpainting
echo "Activated virtual environment:"
echo $CONDA_DEFAULT_ENV
echo "bash ${@:1}"
bash "${@:1}"

cd /home/xogay/AOT-GAN-for-Inpainting/src
python train.py --dir_image "/home/xogay/AOT-GAN-for-Inpainting/src/data/new/gt" --dir_mask "/home/xogay/AOT-GAN-for-Inpainting/src/data/mask" --data_train "" --data_test "" --mask_type ""  --image_size 128  --batch_size 16 --print_every 1000  --save_every 1000 --pre_train "./experiments/aotgan_128/" --resume

echo FINISHED AT $(date)