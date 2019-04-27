import argparse

from fetal_net.data import open_data_file

parser = argparse.ArgumentParser()
parser.add_argument("--data1_path", help="specifies model path",
                    type=str, required=True)
parser.add_argument("--data2_path", help="specifies model path",
                    type=str, required=True)
opts = parser.parse_args()

ids_1 = open_data_file(opts.data1_path).root.subject_ids
ids_2 = open_data_file(opts.data2_path).root.subject_ids

print(all([i1 == i2 for i1, i2 in zip(ids_1, ids_2)]))
