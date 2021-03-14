#!/bin/bash

# Installation des drivers de 


# Installation depuis les repos de Ubuntu // Pas de version récente
$ sudo apt update
$ sudo apt install nvidia-cuda-toolkit




#Installation depuis le repo de NVIDIA // Version plus récente

##wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run

# Changement de la version de Gcc/Clang pour eviter les problème d'installtion sur Ubuntu 20.04
##sudo apt -y install gcc-8 g++-8
##sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
##sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

#Installation de cuda 10.2 version de base 
##sudo sh cuda_10.2.89_440.33.01_linux.run


#L'installation de CUDA Toolkit requiert l'acceptation des termes de de la license EULA.
#En saisissant "accept" 

# Installation de CuDNN // A télécharger manuellement avec son compte
tar -xzvf cudnn-10.1-linux-x64-v8.0.5.39.tgz cuda/
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
