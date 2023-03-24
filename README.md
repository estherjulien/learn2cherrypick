# Code for the paper:
> **Constructing Phylogenetic Networks via Cherry Picking and Machine Learning**  
> *Giulia Bernardini, Leo van Iersel, Esther Julien and Leen Stougie*

To run the code, the following packages have to be installed: `numpy`, `pandas`, `scikit-learn`, `networkx`, and 
`joblib`.

### Cherry picking heuristic (CPH)
The CPH is implemented in `CPH.py`. To experiment with this file,

In the file `test_data_gen_rets.py`, tree sets used for testing can be generated.

### Generate training data
The code for initializing and updating the features can be found in `Features.py`.

For training the random forest, we first have to generate training data. 
The code for this is given in `train_data_gen.py`. 

### Train random forest
In the folder `LearningCherries` you can find the code for training a random forest. 
In the folder `data/RFModels`, there are some trained random forests in `joblib` format.
