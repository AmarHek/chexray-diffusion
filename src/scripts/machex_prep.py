import json
import os


if __name__ == "__main__":
    # machex files were copied from LS6 Server to here and have wrong absolute paths in index.json that way
    # Easiest solution: Iterate through all json-files and replace the paths

    old_root = "/archive/datasets/machex/"
    root_path = "/scratch/hekalo/Datasets/machex/"

    # get all datasets like this, since os.walk would take too long
    datasets = os.listdir(root_path)

    # iterate through all datasets
    for dataset in datasets:
        # get original data
        with open(os.path.join(root_path, dataset, "index.json"), "r") as f:
            data = json.load(f)

        # replace every path in data
        for key in data.keys():
            data[key]['path'] = data[key]['path'].replace(old_root, root_path)

        # write to index.json
        with open(os.path.join(root_path, dataset, "index.json"), "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)