import os

from unet3d.prediction import run_validation_cases
import argparse

def main(config):
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose evaluation model.')
    parser.add_argument('--model', type=str, default='isensee2017', required=False,
                        help='Choose model [unet3d, isensee2017]')
    args = parser.parse_args()
    if args.model == 'unet3d':
        import train
        config = train.config
    elif args.model == 'isensee2017':
        import train_isensee2017
        config = train_isensee2017.config
    else:
        raise Exception('Unknown model')

    main(config)
