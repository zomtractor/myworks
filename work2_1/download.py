import yaml

from utils import MinIOHelper

with open('config.yaml', 'r') as config:
    opt = yaml.safe_load(config)
minio_helper = MinIOHelper(**opt['MINIO'])
minio_helper.download_bucket()
