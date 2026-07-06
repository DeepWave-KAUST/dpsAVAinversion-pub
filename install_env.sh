#!/bin/bash
# 
# Installer for diffseisavo
# 
# Run: ./install_env.sh
# 
# F. Brandolin, 22/05/2025

echo 'Creating diffseis environment'

# create conda env
conda env create -f environment.yml
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate diffseisavo
conda env list
echo 'Created and activated environment:' $(which python)

# check torch and diffusers work as expected
echo 'Checking torch version and running a command...'
python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda"))'
echo 'Checking diffusers version and running a command...'
python -c 'import diffusers; print(diffusers.__version__)'

echo 'Done!'