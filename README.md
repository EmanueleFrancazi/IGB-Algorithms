# Initial Guessing Bias: How Untrained Networks Favor Some Classes

This repository contains the code necessary to reproduce the results presented in the paper:

**Title:** [Initial Guessing Bias: How Untrained Networks Favor Some Classes](https://arxiv.org/pdf/2306.00809) \
The necessary modules to be installed with their versions can be found in the `requirements.txt` file.

There are two folders:
- **Initialization** contains the codes for reproducing the initialization experiments and collecting the corresponding statistics (for example, to reproduce the characteristic histograms that indicate (or not) the presence of IGB).
- **Dynamics**, on the other hand, contains the codes for tracking the dynamics and statistics related to performance trends, to be compared with those related to IGB.


## Structure (**Initialization**)

In order to ensure clearer readability the code is divided into 2 scripts: 
* `MainBlock.py` : code on which the simulation runs; it defines the framework of the program and the flow of executed commands.
* `CodeBlocks.py` : a secondary library, called from `MainBlock.py` that implements through methods within it all the functions that `MainBlock.py` needs.\
The program is launched by a ,command within the bash script `PythonRunManager.sh`. Within the script you can select some parameters/flags that will be given as input to `MainBlock.py`. After properly setting the initial parameters (see later sections) the simulations can then be started with the command (from terminal):\
`./PythonRunManager.sh i1 i2` .\
`i1` and `i2` are two integer values such that `i1`<`i2`. With this command we begin the serial execution of `i2`-`i2`+1 replicas of the desired simulation. Specifically, each replica is identified by a numeric index between `i1` and `i2`.
The data for each replica, associated with a given set of parameters, are loaded, during the course of the simulation, into a folder that has the index of the replica in its name.

### Interaction between `MainBlock.py` and `CodeBlocks.py`
The methods in `CodeBlocks.py`, called by `MainBlock.py` during the simulation, are enclosed in classes.
The simulation starts by setting and defining some essential variables. That done, the DNN is generated as an instance of a class, through the command:\
`NetInstance = CodeBlocks.Bricks(params)`.\
The `Bricks(params)` class has to be interpreted as the bricks that will be used to make up the main code; it contains all the blocks of code that performs simple tasks. The general structure is as follows:\
Inside class `Bricks` we instantiate one of the Net classes. So, in this case, we will not use class inheritance but only class composition: we don't use the previous class as super class but simply call them creating an instance inside the class itself. Each of the Net classes inherits the class NetVariables (where important measures are stored)
_________________________________________
**Notes for newcomers in python:**
                
Inheritance is used where a class wants to derive the nature of parent class and then modify or extend the functionality of it. 
Inheritance will extend the functionality with extra features allows overriding of methods, but in the case of Composition, we can only use that class we can not modify or extend the functionality of it. It will not provide extra features.\
**Warning:** you cannot define a method that explicitly take as input one of the instance variables (variables defined in the class); it will not modify the variable value. 
Instead if you perform a class composition as done for NetVariables you can give the variable there defined as input and effectively modify them.                                
_________________________________________


# Running Pipeline
## Bash script

As mentioned, the simulation is started through a bash script (`PythonRunManager.sh`). Within that script, some parameters are set. Specifically:
* **FolderName** : is the name of the folder that will contain all the results of the execution.
* **Dataset** : parameter that identifies the dataset to be used; at present the code accepts either CIFAR10 or GaussBlob as dataset; the latter is a dataset of Gaussian blobs whose elements have the same extention of CIFAR10 images. To include other datasets (e.g. MNIST) some small changes are necessary because of the different data format.
* **Architecture** : parameter that identifies the network to be used for the simulation: some option already available (see "DEFINE NN ARCHITECTURE" in CodeBlocks.py). Including an arbitrary architecture is very simple; just define the corresponding class and a name that identifies it as a parameter, following the example of the networks already present.
* **DataFolder** : Path to the folder that contains the dataset to be used in the simulation
* **LR** : the learning rate that will be used. It can be a single value or a set of values (which will be given one after the other)
* **BS** : the batch size that will be used. Can be a single value or a set of values (which will be given one after another)
* **GF** : This parameter sets the block size, for groupings operated in group norm layers. It can be a single value or a set of values (which will be given one after the other)
* **DP** : Dropout probability. This parameter sets the probability of zeroing entries across dropout layers. It can be a single value or a set of values (which will be given one after the other) 

For each of the above parameters, it is also possible to select more than one value. In this case, `i2`-`i2`+1 runs will be performed sequentially for each combination of the chosen parameters. For each run, the simulation is started, from the bash script, through the command: \
`python3 MainBlock.py $i $FolderName $Dataset $Architecture $DataFolder $LR $BS $GF $DP` \
The `MainBlock.py` script is thus called.

## `MainBlock.py`
The code starts with an initial block, where some general parameters are defined (number of epochs, any changes in dataset composition, the algorithm to be used, seed initialization, ..). To facilitate the connection with CodeBlocks.py we define a `params` dict where we save all the parameters that we want to be able to access also from "CodeBlocks.py". The network instance is then defined, as explained above, and the program is then started. 

### Reproducibility and Initialization: Random seed
Immediately after importing the modules into `MainBlock.py`
we proceed to initialize the random seeds. Note that initialization must be performed on all libraries that use pseudo-random number generators (in our case numpy, random, torch). 
The operation of fixing the seed for a given simulation is a delicate operation since a wrong choice could create an undesirable correlation between random variables generated in independent simulations. 
The following two lines fix the seed: 

```
    t = int( time.time() * 1000.0 )
    seed = ((t & 0xff0000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x0000ff) << 24)   
```
Python time method `time()` returns the time as a floating point number expressed in seconds since the epoch, in UTC. This value is then amplified. Finally, the bit order is reversed so as to reduce the dependence on the least significant bits, further increasing the distance between similar values (more details are given directly in the code, as a comment, immediately after initialization).
The resulting value is then used as a seed for initialization.
The seed is then saved within a file and printed out, so that the simulation can be easily reproduced if required.



### Logging on server
to more easily monitor the runs and their results the code automatically saves logs of relevant metrics on some server which can then be accessed at any time to check the status of the simulation.
Specifically, simulation results will be available in: 
* Tensorboard: no logging is required for such a server. for more information on using tensorboard see [How to use TensorBoard with PyTorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) 
* Wandb: you can access the server by creating a new account or through accounts from other portals (github, google,...). For more details see, for example 
[W&B-Getting Started with PyTorch ](https://docs.wandb.ai/guides/integrations/pytorch) [Intro to Pytorch with W&B ](https://wandb.ai/site/articles/intro-to-pytorch-with-wandb) 

