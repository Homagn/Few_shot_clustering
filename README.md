Whats this about ?

Detailed code on applying a novel few shot clustering technique (EM style) to cluster images using very few actual labels (few shot clustering)
State of the art accuracy acheieved in ImageNet 5-way 5-shot
Demonstrated application in few shot building occupancy detection

Published paper here -> https://arxiv.org/abs/2008.05654


*Complete Dataset*
See Additional_datasets.txt 


*To run*
================================
python main.py 

(all the parameters of the code present in the top few lines of main.py, explained with comments)


*Dependencies:*
================================
pytorch

opencv

numpy


*Dataset directory structure*
=====================================

Directories need to be created :

data/labeled/0

	    /1
	    
...

...

	    /n
	    
(the few labeled images that you have)

(depending upon number of classes present in the few shot learning problem)

data/unlabeled/ 

(dump all the unlabeled images you want to cluster here)

data/validation/0

...

	       /n
	       
(same structure as data/labaled, this folder images used by the algorithm to track convergence progress if you dont have enough annotations for this folder, just comment out the validate() function in main.py)

data/model_pred/0

...

	       /n
	       
(same structure as data/labeled, here the model will store the clustering results in respective folders as the EM algorithm progresses)
