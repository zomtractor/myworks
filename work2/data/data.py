import os

from data import LocalDataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderTrain,RealtimeDataLoaderTrain


def get_training_data(conf,ps=256):

    if(conf['MODE']=='local'):
        assert os.path.exists(conf['LOCAL_DIR'])
        return LocalDataLoaderTrain(conf['LOCAL_DIR'], patch_size=ps, length=conf['LENGTH'])
    else:
        return RealtimeDataLoaderTrain(conf,ps)
    # return DataLoaderTrain(rgb_dir, img_options,length=10000)


def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)


def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)
