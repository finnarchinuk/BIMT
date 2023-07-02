# BIMT
This is an implementation of "Seeing is Believing: Brain Inspired Modular Training" in Tensorflow.

The original paper (with pytorch code) is available here:
https://kindxiaoming.github.io/pdfs/BIMT.pdf

The BIMT class (and plotting tool) is available in BIMT.py for use and modification. The notebook shows the expected results and some explanation as to how I went about implementing the swapping function.

One interesting change is that my implementation can allow momentum based optimizers (ie, Adam), which required some unusual engineering. It was resolved by reinitializing the optimizer at the beginning of the epoch then applying tf.function to the training step ONCE, otherwise it will throw errors about creating variables in the optimizer multiple times. It may be more efficient to zero out the momentum, but I couldn't figure it out and this provided the speed up I was looking for.

The selection of hyperparameters is very challenging, and I would strongly suggest incorporating some kind of tuning or external logging program (eg, wandb, tensorboard, comet).

Enjoy!

***
An image from the notebook showing a demonstration of how the position swapping is done. More information in the notebook.
![Swapping Weights](https://github.com/finnarchinuk/BIMT/blob/main/Swapped%20Weights.png)
