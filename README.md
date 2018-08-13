### Training
* Install dependencies: 
```
nibabel,
keras,
pytables,
nilearn,
SimpleITK,
nipype
```

* Data Normalization:
(nipype is required for preprocessing only)

Install [ANTs N4BiasFieldCorrection](https://github.com/stnava/ANTs/releases) and add the location of the ANTs 
binaries to the PATH environmental variable.

Add the repository directory to the ```PYTONPATH``` system variable:
```
$ export PYTHONPATH=${PWD}:$PYTHONPATH
```
Convert the data to nifti format and perform image wise normalization and correction:

cd into the brats subdirectory:
```
$ cd brats
```
Import the conversion function and run the preprocessing:
```
$ python
>>> from preprocess import convert_brats_data
>>> convert_brats_data("data/original", "data/preprocessed")
```

* To run training::
```
$ python train_fetal.py
```

### Write prediction images from the validation data
In the training above, part of the data was held out for validation purposes. 
To write the predicted label maps to file:
```
$ python predict.py
```
The predictions will be written in the ```prediction``` folder along with the input data and ground truth labels for 
comparison.
