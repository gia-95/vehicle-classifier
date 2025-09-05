# Vehicle-classifier MAKEFILE

PROJECT := Vehicle-classifier

#########################
# SET PROJECT VARIABLES
#########################
#export PROJ_BASE_DIR = /Users/gmarini/dev/vehicle-classifier
export PROJ_BASE_DIR = c:\Users\mauro\dev\vehicle-classifier 
export DVC_DATA_DIR = $(addsuffix \dvc\data, $(PROJ_BASE_DIR))
export DVC_MODELS_DIR = $(addsuffix \dvc\models, $(PROJ_BASE_DIR))
export WRITER_DIR = $(addsuffix \logs, $(PROJ_BASE_DIR))


# Check the operating system
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
else
    DETECTED_OS := $(shell uname -s)
endif


help:
	@echo '-----------------------------------------------'
	@echo 'Project $(PROJECT) help'
	@echo 'Make running on $(DETECTED_OS)'
	@echo 'User: $(DETECTED_USER)'
	@echo '-----------------------------------------------'
	@echo 'help         : shows this page'
	@echo 'install-cpu  : pip install all dpendencies for CPU systems'
	@echo 'install-cuda : pip install all dpendencies for CUDA systems'
	@echo 'clean        : !!!deletes!!! ./logs contents'
	@echo 'tb           : runs tensorboard on ./logs directory'
	@echo 'test         : runs python test.py'
	@echo 'train        : runs train.py'
	@echo '-----------------------------------------------'
install-cpu:
	pip install -r requirements.txt
install-cuda:
	pip install -r requirements.txt
	pip uninstall torch torchvision
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
run:
	echo "Run Vehicle Classisifer"
	python
clean:
	@echo "Cleaning Logs Directory..."
	@python $(addsuffix \tools\delete_directory_content.py, $(PROJ_BASE_DIR)) $(WRITER_DIR) 
build:
	echo "Build...."
tb:
	tensorboard --logdir $(WRITER_DIR)
train:
	python ./src/LeNet/train_mauro.py --lr 0.001 --epochs 10
	python ./src/LeNet/train_mauro.py --lr 0.01 --epochs 10
test:
	python ./src/test.py
