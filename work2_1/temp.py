import torch

from model import MyNet2, UBlock
from torchsummary import summary

from utils import MinIOHelper
import os

if __name__ == '__main__':
    minio_helper = MinIOHelper(
            endpoint='47.95.21.85:9000',
            access_key='admin',
            secret_key='admin666',
            secure=False
    )
    # minio_helper.upload_file("checkpoints/DeFlare/models/123.txt")
    # minio_helper.download_directory("DeFlare/train_logs")
    minio_helper.upload_directory( "checkpoints/DeFlare/train_logs")

