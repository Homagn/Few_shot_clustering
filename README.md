Whats this about ?

Detailed code on applying a novel few shot clustering technique (EM style) to identigy occupied/unoccupied regions in rooms using as few supervised examples as possible with low quality images.
Published paper here -> https://arxiv.org/abs/2008.05654


*Complete Dataset*
See Additional_datasets.txt 


*To run*
================================
python main.py 


*Environment instructions:*
================================
conda create -n oldkeras python=3.6

conda activate oldkeras

pip install tensorflow==1.14.0

pip install keras==2.0.4 

pip install Pillow 

pip install dill

conda install scikit-learn

pip install matplotlib

*Dataset directory structure*
=====================================

1. Directories need to be created :
Instances/
	cal_data - contains subfolder occupied and unoccupied 
	(This folder contains the data used to train the one shot classifier)- data goes as x_train and y_train (in main.py code)
	(During start of training cal_data/ can just be an empty folder containing the two empty folders occupied/ and unoccupied/)
	(However during training you can keep checking its images to see how good the classifier is able to seperate image instances)

	sup_set - contains subfolder occupied and unoccupied
	(After training, when in inference, pairings are formed from data in this folder to the test image to get output of the classifier)
	- data goes as x_val and y_val (in main.py code)
	(During testing the effectiveness of the algorithm, you may actually do the hard work and seperate instances into occupied/ and unoccupied/ inside this folder by hand, accurately. One advantage this will provide is that you can monitor the accuracies perfectly)
	(However when actually training the algorithm, sup_set represents all the unlabeled data that we want to label, so you can just randomly dump images in an equal number in the occupied/ and unoccupied/ folder irrespective of whether they are correct classes or not)


	test_set - contains subfolder occupied and unoccupied
	(This folder contains all the data you have annotated as occupied or unoccupied) - data goes as x_test and y_test (in main.py code). The algorithm is powerful enough to start training with only as low as 5 pairs of annotated images (occupied and unoccupied)



test_results/
	 - contains subfolder occupied and unoccupied
	(After running inference on test_set, the results are stored here in the respective classes folders. 
	Images are saved with the corresponding confidence in classification as the name)

weights/
	Stores the saved model weights

The code in its first run creates several obj files in the main directory for easy loading of the dataset again later on

2. Clean all the .obj files if your dataset has changed.

3. Run main.py (It first creates preprocessed pickle files for x_train, x_val and x_test for easy loading later on)