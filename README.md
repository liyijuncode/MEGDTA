# MEGDTA
Multi-modal drug-target affinity prediction based on protein three-dimensional structure and ensemble graph neural network
# Requirements
python 3.8.12, pytorch 1.10.1, numpy 1.24.4, pandas 1.4.4, torch_geometrci 2.4.0, scipy 1.10.1
# Model training
1. Create a virtual environment  
`conda create -n MEGDTA python=3.8`  
`conda activate MEGDTA`  
2. Install required packages  
`conda install numpy, scipy, pandas, torch, torch_geometric`
3. Clone the repository  
`git clone https://github.com/`  
`cd MEGDTA/`
4. Train the model and obtain prediction results using the command  
`cd MEGDTA/` Eavigate to the specified directory  
`python test.py` Train on all datasets and obtain the prediction results   
`python test.py --datasets davis` Choose to train on the davis dataset and obtain the results  
`python test.py --datasets kiba` Choose to train on the kiba dataset and obtain the results    
`python test.py --datasets metz` Choose to train on the metz dataset and obtain the results  
The training results will be saved in the `results/` folder, if a GPU is available, you can use command `python test.py --datasets name --gpu GPU id to use (e.g., 0, 1, 2, 3, 4)'` to select the GPU device for model training and prediction
After the run finishes, the data containing the best predictions and true values corresponding to the optimal MSE for each fold will be saved with the filename `fold_{fold}_best_predictions_{dataset}.csv`  
# Data  
Please check the data under the `data/` directory









