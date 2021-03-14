# Installation

Pour la detection, nous nous servirons le long de ce projet de l'outil Detecrtron2 de FAIR basé sur PyTorch, c'est l'outil est le plus rapide pour le moment pour entrainer et configurer des modèles, avec un grand choix d'algorithmes de detection d'objet.

Ensuite, nous utiliserons ensuite ROS Noestic pour detecter les objets à base des scripts python. "TorchScript" n'étant pas encore disponible pour C++ , une alternative serait d'utiliser "Onnx"

/////////////////// Installation de Detectron 2 pour GPU ///////////////
## Pre-requis

### Installation des drivers NVIDIA
#### Installation de CUDA Toolkit
Installer la version correspondante au système depuis :
" https://developer.nvidia.com/cuda-downloads "

#### Installation de CuDNN
1 - Nécessite la création d'un compte & téléchargement des archives depuis : 
" https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux "

2 - Acceptation des termes.

3 - Installation :

```
tar -xzvf cudnn-10.1-linux-x64-v8.0.5.39.tgz cuda/
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn* 
```
#### Installation rapide depuis les repos d'Ubuntu ! Pas de version recente
```
sudo apt update
sudo apt install nvidia-cuda-toolkit
```
pq : Script bash d'installation depuis les repos disponible via :

``` 
cd /dependencies/cuda_install/
sudo bash /dependencies/cuda_install/install_cuda.sh
```

### Installation de TorchVision
1 - Choisir sa version de CUDA sur : https://pytorch.org/get-started/locally/ 
2 - Puis installer via conda 
 ps :  https://github.com/pytorch/pytorch#from-source pour un build from source "LibTorch"

### librairies complementaire à installer selon le script utilisé
A installer via conda ou pip :
{numpy, openCV, cython, Scipy, COCOAPI (optionel)} 


## Installation de Detectron 2
```
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

/////////////////// Installation de ROS noetic ///////////////

1 - ajout des repos sur ubuntu
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```

2 - initialisation de la clé pour acceder au serveur
```
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
```

3 - Update des repos, puis installation : 
```
sudo apt update
sudo apt install ros-noetic-desktop-full 
```
4 - Installation de l'environnement :
```
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
```

5 - Sourcing de l'environnement :
```
source /opt/ros/noetic/setup.bash
source devel/setup.bash
```

ps : Utliser la version en phase de deploiement

<<<<<<< HEAD
=======
/////////////////// Installation usb-cam pour ROS noetic ///////////////
# Installation de usb-camera sur ROS

>>>>>>> 741751523c45e7f134195c173b209b8bf1b5a479
```
cd ~/catkin_ws/src
git clone https://github.com/ros-drivers/usb_cam.git
cd ~/catkin_ws 
catkin_make
```
# Installation de cvBridge sur ROS pour utiliser openCV

```
cd ~/catkin_ws/src
git clone https://github.com/ros-perception/vision_opencv.git
cd ~/catkin_ws 
catkin_make
```

