Lowering learning rate seems to have a good effect on accuracy. We also see clear difference with a small penalization on weight in term of complexity reduction.
Even with the increase of accuracy, we are still far from potential results obtained with a similar penalization on a subnetwork. 
I propose not investigate more on combining weight decay with size regularization, but the following experiments :
	- Lower learning rate to see wheter we can approach similar accuracy and keep a sufficient reduction in complexity. 
	- Compare possible results obtainable by first train with reduction and then only with weight decay
	- Observe if lower training time for more experiments would affect a lot results or not.
	- See if training without reconstruct a network affect sparsity along with the accuracy (Remind talk with Patrick)
