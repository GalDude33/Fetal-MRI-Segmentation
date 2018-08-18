import os

from fetal_net.prediction import run_validation_cases

from train_fetal import config

overlap_factor = 1


def main():
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir,
                         overlap_factor=overlap_factor)


if __name__ == "__main__":
    main()
