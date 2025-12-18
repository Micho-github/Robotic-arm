import os


def dataset_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"dataset")


def check_dataset(classes, data_dir):
    data_dir = data_dir or dataset_path()
    if not data_dir or not os.path.exists(data_dir):
        return False

    split_train = os.path.join(data_dir, "train")
    split_val = os.path.join(data_dir, "val")
    has_split = os.path.isdir(split_train) and os.path.isdir(split_val)

    if has_split:
        return (
            all(os.path.isdir(os.path.join(split_train, c)) for c in classes)
            and all(os.path.isdir(os.path.join(split_val, c)) for c in classes)
        )

    return all(os.path.isdir(os.path.join(data_dir, c)) for c in classes)
