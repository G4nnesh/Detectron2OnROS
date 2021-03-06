#! bin/bash

# ajout des repos sur ubuntu
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

#initialisation de la clé pour acceder au serveur
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# installation : update apt avant !
sudo apt install ros-noetic-desktop-full #utliser la version bare Bones sur le Kit

#### Ne pas oublier d'installer les packages nécessaire manuellement ! ###

# Ajouter l'environnement au terminal
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
