# README

Detectron2OnROS : a Detectron2 package for ROS
=======
El Mehdi CHOUHAM in collaboration with CRISTAL laboratory
---

# Projet
This project is a poursuite of the project initiated by Baas Simon & Meinas Julien in collaboration with the CRISTAL Laboratory using TensorFlow, for more information on their project: 
https://github.com/simonbaas-gif/PJI-BAAS-MEINAS

In this project is a project detection package for ROS Noetic using FAIR's Detecrtron2 tool based on PyTorch, the fastest tool at the moment to train and configure models, with a large choice of object detection algorithms.
This package will supply other device nodes with the needed information to perform road obstacles avoidance.


## dev type install (not recommanded)
@devs If you have the same system carateristics as this project developper, follow install.md steps
if you don't have the same architecture please ignore this step !

## Prequisites (source installation recommanded)
# for your device to use gpu install
cuDNN/ Cuda Toolkit

# Python requirements
pyTorch/ libtorch
openCV 
detectron2

# Linux requirements
ROS==ROS NOETIC
pyhton-is-python3


## Configuration

- Use ./configure_predictor/tools/labelme2coco.py to generate a COCO format json label file
- Fine-tune the model to your classes and evaluate it on the notebook jupyter ./configure_predictor/tools/train_&_evaluate.ipynb

if no model was suppplied, faster rcnn is use as default

## Create ROS package

- Export your model on the script in ./detection_ROS/catkin_ws/src/detection_pkg/scripts/detector_from_input.py
- Go back to ./detection_ROS/catkin_ws and rebuild with `catkin_make`

## Run Ros detection package

- source your environnement with `source devel/setup.bash`
- launch package with : `roskaunch detection_pkg detection_pkg.launch`

---

## TODO for obstacles avoidance
1 - Create a movement law for the car<br/>
2 - Use the 'obstacles' topic msg to command the car


## Default model


## Caractèristiques et Suggestions

### Dataset
coco entrainé sur coco pour plus de précision (volumineuse mais plein d'alternatives diponibles)
=======
Ce projet est une suite du projet initier par Baas Simon & Meinas Julien en collaboration avec le Laboratoire CRISTAL utilisant TensorFlow, pour plus d'information sur leurs projet : 
https://github.com/simonbaas-gif/PJI-BAAS-MEINAS

Dans ce projet on introduit l'utilisation de l'outil Detecrtron2 de FAIR basé sur PyTorch, c'est l'outil est le plus rapide pour le moment pour entrainer et configurer des modèles, avec un grand choix d'algorithmes de detection d'objet.

On implemente ensuite le script obtenue sur le middleware ROS noetic, en créant une node qui detectent les objects.


## Installation du projet

Les étapes concernant l'installation des drives NVIDIA, Detectron2 et ses dependences ainsi que ROS Noetic et les packages/ librairies utiles sont disponibles dans install.md.

## Les différents scripts du projet

- PredictorFromModel.py

Script de configuration du modèle : Crée une classe du prédicteur basé sur le modèle entrainé vie le notebook jupyter disponible dans tool/configuration_model.ipynb

- objectPredictor.py

Script principal : Lance la détection d'objets sur les images de test présentes dans le répertoire test_images et affiche la sortie en cpu ou vua les gpus.


## Execution des scripts
- Utiliser -help pour avoir plus d'information sur l'execution de objectPredictor.py
    exemple pour un flux camera affiché sur une fenêtre : python objectPredictor.py --camera <port de la camera>

## Installation des librairies

Le projet nécessite l'installation de différentes librairies python comme décrit dans install.md. L'installation via conda est recommandée !

---

## Procédure suivie
1 - Entrainement et configuration du modèle sur le notebook : detection_Detectron2/tool/configuration_model.ipynb

2 - Une fois le modèle fixé, on le copie dans le zone réservé dans le script PredictorFromModel.py

3 - Implémentation des scripts dans l'environnement ROS: /detection_ROS/catkin_ws/scripts 
!Puis, affecter un droit d'éxecution au script (+x) !

4 - Création des packages et des nodes via rospy #TODO

5 - Récuperation du Topic et contrôle du mouvement #TODO

## Structure du projet 
├── dependencies
│   ├── cuda_install #Dossier d'extraction de cudnn et cuda toolkit + script bash d'installation
│   │   └── cuda
│   ├── libTorch_install #Dossier d'extraction de libtorch + script bash d'installation mais TorchVision est recommendé
│   │   ├── libtorch
│   │   └── pytorch
│   └── ros_install
├── detection_Detectron2 # Dossier des scripts de detection Detectron
│   ├── configs # Configuration possibles
│   ├── modelCheckpoints # Configuration possibles
│   ├── output # Dossier des sauvegardes de la sortie des test si choisie choisie
│   ├── __pycache__
│   ├── test_images # Jeu de tests. (pour comparer à la preière version du projet)
│   │   └── result
│   └── tool #Notebook Jupyter et dépendences pour configurer, tester et voir les métriques du modèle avant de l'implementer
└── detection_ROS # Dossier de la partie Middleware
    └── catkin_ws # Dossier du WorkSpace
        ├── build
        ├── devel # Installation des packages compilé
        ├── libuvc
        ├── scripts # scripts utilsés
        └── src # packages

## TODO 
1 - Création d'une node ROS noetic basé sur le script
2 - Connexion des bus du contrôleur recemment reçus du constructeur et contrôle


## caractèristiques et suggestions

### Dataset
coco entrainé sur coco pour plus de précision (volumineuse mais plein d'alternatives diponible)


### Dataset pour tester le projet

/test_image

### Traitement d'images

Traitement direct par les utilités de detectron2

### Dataset interessantes 

https://bdd-data.berkeley.edu/ 

Idéale pour notre utilisation mais fichier des annotations non gratuit !

=======
### Création des Modèles

On initialise les poids d'après n'importe quel modèle déjà existant puis en l'entraine.
On configure ensuite le node cfg pour modifier le seuil, le cpu/ les gpus utilisés pour la prédiction /entrainement, les classes ....

#### Modèle final

On préviligie ici de garder un modèle entrainer par la méthode mask rcnn R-CNN  utilisant un backbone ResNet.

#### D'autres Modèles utilisant un backbone ResNET
https://reposhub.com/python/deep-learning/zhanghang1989-detectron2-ResNeSt.html

### Réferences et liens utiles : 

#### GitHub Lidar_Obstacle_Detection

https://github.com/udacity/SFND_Lidar_Obstacle_Detection
https://github.com/studian/SFND_P1_Lidar_Obstacle_Detection

- https://github.com/ajimenezh/self-driving-car-obstacle-detector

- https://github.com/alirezaasvadi/ObstacleDetection

---

### Documents intéressants pour le projet :

#### Le modèle ONNX

https://docs.microsoft.com/fr-fr/dotnet/machine-learning/tutorials/object-detection-onnx

ONNX (Open Neural Network Exchange) est un format open source pour les modèles IA. ONNX prend en charge l'interopérabilité entre les frameworks. On peut donc entraîner un modèle dans l'un des nombreux frameworks (K, ML.NET, PyTorch, ...). On peut donc par exemple passer de PyTorch à ML.NET en consommant le modèle ONNX.

##### Le site ONNX
https://onnx.ai/

A noter que le Framework/Converter supporte PyTorch. Et est donc plus convenable car TorchScript n'est pas supporté par Detectron2
En fonction du framework/outil utilisé, la procédure pour convertir le projet au format ONNX est différente. 
Pour plus d'informations : https://github.com/onnx/tutorials#converting-to-onnx-format


##### Ségmentation Panopti
https://kharshit.github.io/blog/2019/10/18/introduction-to-panoptic-segmentation-tutorial

### Création d'une node sur ROS pour la detection via Detectron2 
https://github.com/DavidFernandezChaves/Detectron2_ros
