import yaml
import os
from utils import MinIOHelper

if __name__ == '__main__':

    with open('config.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    minio_helper = MinIOHelper(**opt['MINIO'])
    # minio_helper = MinIOHelper(
    #         endpoint='objectstorageapi.ap-northeast-1.clawcloudrun.com',
    #         access_key='16bqw05c',
    #         secret_key='h8zdm5kg6k9kg26z',
    #         bucket_name="16bqw05c-mywork",
    #         secure=True
    # )
    pth = os.path.join(opt['MODEL']['MODE'],'models', 'model_latest.pth')
    minio_helper.download_file(pth)
    logs = os.path.join(opt['MODEL']['MODE'],'train_logs')
    minio_helper.download_directory(logs)
