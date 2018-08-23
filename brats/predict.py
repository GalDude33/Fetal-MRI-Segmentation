import json
import os

from fetal_net.prediction import run_validation_cases
import argparse


def main(config, overlap_factor=1):
    prediction_dir = os.path.abspath(config['base_dir'])
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir,
                         overlap_factor=overlap_factor,
                         patch_shape=config["patch_shape"]+[config["patch_depth"]],
                         prev_truth_index=config["prev_truth_index"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", help="specifies config dir path",
                        type=str, required=True)
    parser.add_argument("--overlap_factor", help="specifies overlap between prediction patches",
                        type=int, default=1)
    opts = parser.parse_args()

    with open(os.path.join(opts.config_dir, 'config.json')) as f:
        config = json.load(f)

    main(config, opts.overlap_factor)
