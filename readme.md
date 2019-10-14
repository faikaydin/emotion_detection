#### emotion detection

This is the repo for the task. The generated model (model.h5) is a
MobileNet architecture built and trained with Keras.

The model scores a .76 accuracy on your test set and has a size of 13MB.

As the the test set is very small, a really good generalization is needed for a good result.
For this reason Facial Expression Recognition 2013 data set was also used. This dataset is much
larger and helps the model generalize better.

This repo contain files that allows the user train and evaluate.

#### For Training 

There are two types of training, dry and transfer.

For best performance (and the one that generated model.h5) please use 'transfer'

######For Transfer you must specify where the fer dataset is downloaded (runner.py, line 25)

######python runner.py <file-holding-original-data> transfer

######example: python runner.py ~/Downloads/Face_data_split/ transfer


Dry train only trains with the dataset you've sent (which has poor performance)

######python runner.py <file-holding-original-data> dry

######example: python runner.py ~/Downloads/Face_data_split/ transfer

#### For evaluation of test set and generate the predicted version of the test set
###### generate_predictions.py

#### Possible improvement 
I was going to use Google Colab, however it was very buggy on Sunday so I didn't have much time to experiment sadly. 
I ran this above described model over 50 Epochs (~450 min on my CPU).

One possible thing I wanted to try was, after training MobileNet only on the fer2013 data set, I wanted to finetune 
ONLY the decoder with your dataset's training set on a low learning rate. I believe that would've increased the accuracy
but sadly didn't have the time!!! 
