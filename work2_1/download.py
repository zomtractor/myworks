from utils import MinIOHelper

minio_helper = MinIOHelper(
            endpoint='47.95.21.85:9000',
            access_key='admin',
            secret_key='admin666',
            secure=False)
minio_helper.download_bucket()
