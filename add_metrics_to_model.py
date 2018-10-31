from keras import Model
from fetal_net.training import load_old_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", help="specifies model path",
                    type=str, required=True)
opts = parser.parse_args()

from_metric = 'acc'
to_metric = 'binary_accuracy'

model: Model = load_old_model(opts.model_path)
model.compile(optimizer=model.optimizer,
              loss=model.loss,
              metrics=[to_metric if _ == from_metric else _
                       for _ in model.metrics])

model.save(opts.model_path)
