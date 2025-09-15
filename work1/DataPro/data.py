import os

from DataPro import LocalDataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderTrain


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return LocalDataLoaderTrain(rgb_dir, img_options,length=64)
    # return DataLoaderTrain(rgb_dir, img_options,length=10000)


def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)


def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)
