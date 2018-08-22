### Training
* 

* Install Tensorflow & Keras

* Install dependencies: 
pip install -r requirements.txt

* To run training:
Config currently in dict inside brats/train_fetal.py (See [Issue #3](https://github.com/GalDude33/Fetal_Envelope_MRI/issues/3))
```
$ python -m brats.train_fetal
```

### Write prediction images from the validation data
In the training above, part of the data was held out for validation purposes. 
To write the predicted label maps to file:
```
$ python -m brats.predict
```
The predictions will be written in the ```prediction``` folder along with the input data and ground truth labels for 
comparison.
