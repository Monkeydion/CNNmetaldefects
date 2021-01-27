# CNN Metal Defects Image Classification

## In training.py

1. On lines 17-28,only ONE must be uncommented. The rest are commented. 
2. On line 68, change the directory to the corresponding model being trained.
3. Run the code
4. Once training is finished, a matplotlib plot of the training history would popup. Save it. 
5. Also screenshot the log of the training (for reference). 
6. Find the epoch number where the validation accuracy is maximum. If there are several candidates, select the least validation loss

## In predict.py
1. On line 16, select the .h5 file that corresponds to the epoch with the highest validation accuracy. 
2. Run the code
3. Take a screenshot of the confusion matrix

Note: If there is an error with regards to importing the libraries, remove tensorflow and just import keras
