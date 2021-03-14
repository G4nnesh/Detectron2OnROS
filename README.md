# README

Detectron2OnROS : a Detectron2 package for ROS
=======
El Mehdi CHOUHAM in collaboration with CRISTAL laboratory
---

# About
This project is a poursuite of the project initiated by Baas Simon & Meinas Julien in collaboration with the CRISTAL Laboratory using TensorFlow, for more information on their project: 
https://github.com/simonbaas-gif/PJI-BAAS-MEINAS

In this project is a project detection package for ROS Noetic using FAIR's Detecrtron2 tool based on PyTorch, the fastest tool at the moment to train and configure models, with a large choice of object detection algorithms.
This package will supply other device nodes with the needed information to perform road obstacles avoidance.


## dev type install (not recommanded)
@devs If you have the same system carateristics as this project developper, follow install.md steps
if you don't have the same architecture please ignore this step !

# Prequisites (source installation recommanded)
## for your device to use gpu install
cuDNN/ Cuda Toolkit

## Python requirements
pyTorch/ libtorch
openCV 
detectron2

## Linux requirements
ROS==ROS NOETIC
pyhton-is-python3


# Configuration

- Use ./configure_predictor/tools/labelme2coco.py to generate a COCO format json label file
- Fine-tune the model to your classes and evaluate it on the notebook jupyter ./configure_predictor/tools/train_&_evaluate.ipynb

if no model was suppplied, faster rcnn is use as default

# Build ROS package

- Export your model on the script in ./detection_ROS/catkin_ws/src/detection_pkg/scripts/detector_from_input.py
- Go back to ./detection_ROS/catkin_ws and rebuild with `catkin_make`

# Run Ros detection package

- source your environnement with `source devel/setup.bash`
- launch package with : `roskaunch detection_pkg detection_pkg.launch`

---

## TODO for obstacles avoidance
1 - Create a movement law for the car<br/>
2 - Use the 'obstacles' topic msg to command the car


## Default model


## Caractèristiques et Suggestions

## Dataset
=======

entrainé sur coco pour plus de précision (volumineuse mais plein d'alternatives diponible)

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
