### Training 

* Install Tensorflow & Keras

* Install dependencies: 
pip install -r requirements.txt

* To run training:
Config currently in dict inside brats/train_fetal.py
```
$ python -m brats.train_fetal --config_dir <config_dir>
```
Where <config_dir> is the folder containing the training configurations (or will be creatad to)

### Write prediction images from the validation data
In the training above, part of the data was held out for validation purposes. 
To write the predicted label maps to file:
```
$ python -m brats.predict
```
The predictions will be written in the ```prediction``` folder along with the input data and ground truth labels for 
comparison.
