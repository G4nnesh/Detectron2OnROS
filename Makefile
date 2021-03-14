JAUNE = \033[1;33m
ROUGE = \033[0;31m
NC = \033[1;0m

all : install_cuda  install_ros install_libtorch
	@echo "$(JAUNE)Installation de cuda Toolkit, ros Noetic et LibTorch$(NC)"
	@echo "$(ROUGE)/!\ Adapté à l'ordinateur de developpement & nécessite de télécharger cuda et libTorch au préalable ! pour installer sur votre machine suivez les instructions du fichier install.md/!\$(NC)"


@PHONY :
install_cuda :
	@echo "$(JAUNE)Installation de cuda Toolkit$(NC)"
	@echo "$(ROUGE)/!\ Adapté à l'ordinateur de developpement & nécessite de télécharger et extraire cuda Toolkit et cudnn au préalable ! pour installer sur votre machine suivez les instructions du fichier install.md/!\$(NC)"
	$Q cd dependencies/cuda_install/ && sudo bash install_cuda.sh 
install_ros :
	@echo "$(JAUNE)Installation de ROS Noetic$(NC)"
	$Q cd dependencies/ros_install/ && sudo bash install.sh
install_libtorch :
	@echo "$(JAUNE)Installation de libTorch$(NC)"
	@echo "$(ROUGE)/!\ Adapté à l'ordinateur de developpement & nécessite de télécharger et extraire libTorch au préalable ! pour installer sur votre machine suivez les instructions du fichier install.md/!\$(NC)"
	$Q cd /dependencies/libTorch_install/ && sudo bash install_libtorch.sh

clear :
	$Q cd ~/dependencies/cuda_install/
	$Q find . ! -name "install_cuda.sh" -exec rm -r {} \;
	$Q cd ~/dependencies/ros_install/
	$Q find . ! -name "install_cuda.sh" -exec rm -r {} \;
	$Q cd ~/dependencies/libTorch_install/
	$Q find . ! -name "install_cuda.sh" -exec rm -r {} \;