# Assignment CMPUT811
## Start
If you are able to run the test code from tutorial means your environment is setting up correclly.
If not please got to [tutorial page](https://github.com/Lucklyric/theanets-tutorial/blob/master/README.md) to setup your environment.

## Files
* twosensors.csv,twosensors2.csv: Two datasets.
* twosensrosfusion.py: Main script.
* test-trained-model.py: Test script for trained model.
* models/: The directoy that includes all trained models.

## Train the model
* Using CPU
`````
HEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cpu python twosensorsfusion.py
`````
* Using GPU, make sure you have isntalled Cuda (tested with Cuda version 7.5)
`````
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu python twosensorsfusion.py
`````

## Evaluate trained model
`````
python test-trained-model.py <path of trained model> <path of test dataset>
`````

## Report
[Link to the report.](https://github.com/Lucklyric/theanets-tutorial/blob/master/mm811as3-homwork/report.md)
