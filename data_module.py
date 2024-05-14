import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_colon_dataset(stage="train", data_dir="/data2/HistoImages/colon_kbsmc/COLON_MANUAL_512"):
    df = pd.read_csv(os.path.join(data_dir, "{}.csv".format(stage)))
    image_id = df["path"].values
    labels = df["class"].values
    return list(zip(image_id, labels))


def prepare_colon_test2_dataset(stage="test", data_dir="/data2/HistoImages/colon_kbsmc/test_2/colon_45WSIs_1144_08_step05_05"):
    file_list = glob.glob(f"{data_dir}/*/*/*.png")
    label_list = [int(file_path.split("_")[-1].split(".")[0]) - 1 for file_path in file_list]
    return list(zip(file_list, label_list))


def prepare_gastric_dataset(stage="train", csv_dir="/data2/HistoImages/gastric_kbsmc/csv"):
    csv_path = os.path.join(csv_dir, "class4_step10_ds_{}.csv".format(stage))
    df = pd.read_csv(csv_path)
    image_id = df["path"].values
    labels = df["class"].values
    return list(zip(image_id, labels))


def prepare_prostate_harvard_dataset(stage="train", data_dir="/data2/HistoImages/prostate_harvard"):
    def _load_data_info(path_name):
        file_list = glob.glob(path_name)
        label_list = [int(file_path.split("_")[-1].split(".")[0]) for file_path in file_list]
        return list(zip(file_list, label_list))

    if stage == "train":
        data_root_dir = os.path.join(data_dir, "patches_train_750_v0")
        train_set_111 = _load_data_info(f"{data_root_dir}/ZT111*/*.jpg")
        train_set_199 = _load_data_info(f"{data_root_dir}/ZT199*/*.jpg")
        train_set_204 = _load_data_info(f"{data_root_dir}/ZT204*/*.jpg")
        data_pair = train_set_111 + train_set_199 + train_set_204
    elif stage == "valid":
        data_root_dir = os.path.join(data_dir, "patches_validation_750_v0")
        data_pair = _load_data_info(f"{data_root_dir}/ZT76*/*.jpg")
    elif stage == "test":
        data_root_dir = os.path.join(data_dir, "patches_test_750_v0")
        data_pair = _load_data_info(f"{data_root_dir}/patho_1/*/*.jpg")
    else:
        raise NotImplementedError

    return data_pair


def prepare_prostate_ubc_dataset(stage="test", data_dir='/data2/HistoImages/prostate_ubc/prostate_miccai_2019_patches_690_80_step05'):
    """
    prostate_miccai_2019_patches_690_80_step05
    class 0: 1811
    class 2: 7037
    class 3: 11431
    class 4: 292
    1284 BN, 5852 grade 3, 9682 grade 4, and 248 grade 5
    """

    def _split(dataset):  # train val test 80/10/10
        train, rest = train_test_split(dataset, train_size=0.8, shuffle=False, random_state=42)
        valid, test = train_test_split(rest, test_size=0.5, shuffle=False, random_state=42)
        return train, valid, test

    files = glob.glob(f"{data_dir}/*/*.jpg")

    data_class0 = [data for data in files if int(data.split("_")[-1].split(".")[0]) == 0]
    data_class2 = [data for data in files if int(data.split("_")[-1].split(".")[0]) == 2]
    data_class3 = [data for data in files if int(data.split("_")[-1].split(".")[0]) == 3]
    data_class4 = [data for data in files if int(data.split("_")[-1].split(".")[0]) == 4]

    train_data0, validation_data0, test_data0 = _split(data_class0)
    train_data2, validation_data2, test_data2 = _split(data_class2)
    train_data3, validation_data3, test_data3 = _split(data_class3)
    train_data4, validation_data4, test_data4 = _split(data_class4)

    label_dict = {0: 0, 2: 1, 3: 2, 4: 3}

    if stage == "train":
        data_paths = train_data0 + train_data2 + train_data3 + train_data4
    elif stage == "valid":
        data_paths = (validation_data0 + validation_data2 + validation_data3 + validation_data4)
    elif stage == "test":
        data_paths = test_data0 + test_data2 + test_data3 + test_data4
    else:
        raise NotImplementedError

    data_label = [int(path.split(".")[0][-1]) for path in data_paths]
    data_label = [label_dict[k] for k in data_label]
    data_pair = list(zip(data_paths, data_label))

    return data_pair
