!# bin/bash

#install conda preq
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

#install conda
sudo bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh

#download PyTorch Source
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

#install PyTorch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(/root/anaconda3))/../"}
python setup.py install

