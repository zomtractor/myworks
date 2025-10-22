#!/usr/bin/env python3
"""
MinIO文件同步工具
用于与MinIO bucket "mywork" 进行文件同步，保持与本地"checkpoints"文件夹一致
"""

import os
import argparse
import hashlib
import shutil
from pathlib import Path
from minio import Minio
from minio.commonconfig import CopySource
from minio.error import S3Error

from utils import minio_logger


class MinIOHelper:
    def __init__(self, endpoint, access_key, secret_key,bucket_name="mywork", secure=False):
        """
        初始化MinIO客户端

        Args:
            endpoint: MinIO服务器地址 (例如: 'localhost:9000')
            access_key: 访问密钥
            secret_key: 秘密密钥
            secure: 是否使用HTTPS
        """
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name
        self.local_dir = "checkpoints"

        # 确保本地目录存在
        Path(self.local_dir).mkdir(exist_ok=True)

        # 确保bucket存在
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """确保bucket存在，如果不存在则创建"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                minio_logger.info(f"Bucket '{self.bucket_name}' 创建成功")
            else:
                minio_logger.info(f"Bucket '{self.bucket_name}' 已存在")
        except S3Error as e:
            minio_logger.error(f"检查/创建bucket时出错: {e}")
            raise

    def _get_file_md5(self, file_path):
        """计算文件的MD5值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_relative_object_name(self, file_path):
        """获取文件相对于本地目录的对象名称"""
        file_path = Path(file_path).resolve()
        local_dir_path = Path(self.local_dir).resolve()

        if str(file_path).startswith(str(local_dir_path)):
            # 如果文件在checkpoints目录或子目录中，保持相对路径
            object_name = str(file_path.relative_to(local_dir_path))
            # 确保使用正斜杠作为路径分隔符（MinIO使用正斜杠）
            object_name = object_name.replace('\\', '/')
            return object_name
        else:
            # 如果文件不在checkpoints目录中，直接使用文件名
            minio_logger.warning(f"文件不在checkpoints目录中，将使用文件名作为对象名: {file_path.name}")
            return file_path.name

    def _get_local_path_from_object(self, object_name):
        """根据对象名称获取本地文件路径"""
        return Path(self.local_dir) / object_name

    def download_bucket(self):
        """
        下载整个bucket的文件到本地checkpoints文件夹
        保持目录结构一致
        """
        try:
            minio_logger.info(f"开始下载bucket '{self.bucket_name}' 到本地目录 '{self.local_dir}'")

            # 获取bucket中的所有对象
            objects = self.client.list_objects(self.bucket_name, recursive=True)
            downloaded_count = 0

            for obj in objects:
                # 构建本地文件路径
                local_path = self._get_local_path_from_object(obj.object_name)

                # 确保本地目录存在
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # 下载文件
                self.client.fget_object(
                    self.bucket_name,
                    obj.object_name,
                    str(local_path)
                )
                downloaded_count += 1
                minio_logger.info(f"下载文件: {obj.object_name}")

            minio_logger.info(f"下载完成! 共下载 {downloaded_count} 个文件")

        except S3Error as e:
            minio_logger.error(f"下载bucket时出错: {e}")
            raise

    def download_file(self, remote_object_name, local_path=None):
        """
        下载单个文件到本地checkpoints的相应位置

        Args:
            remote_object_name: 远程对象的名称（在bucket中的路径）
            local_path: 本地文件路径，如果为None则自动根据对象名称确定
        """
        try:
            if local_path is None:
                # 自动确定本地路径
                local_path = self._get_local_path_from_object(remote_object_name)
            else:
                local_path = Path(local_path)

            # 确保本地目录存在
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # 下载文件
            self.client.fget_object(
                self.bucket_name,
                remote_object_name.replace('\\', '/'),
                str(local_path)
            )

            minio_logger.info(f"文件下载成功: {remote_object_name} -> {local_path}")
            return True

        except S3Error as e:
            minio_logger.error(f"下载文件时出错: {e}")
            return False

    def upload_file(self, file_path):
        """
        上传单个文件到MinIO，保持目录结构

        Args:
            file_path: 要上传的本地文件路径
        """
        try:
            # 检查文件是否存在
            if not os.path.isfile(file_path):
                minio_logger.error(f"文件不存在: {file_path}")
                return False

            # 获取对象名称
            object_name = self._get_relative_object_name(file_path)

            minio_logger.info(f"上传文件: {file_path} -> {object_name}")

            # 上传文件（强制替换）
            self.client.fput_object(
                self.bucket_name,
                object_name,
                str(file_path)
            )

            minio_logger.info(f"文件上传成功: {object_name}")
            return True

        except S3Error as e:
            minio_logger.error(f"上传文件时出错: {e}")
            return False

    def copy_file_local_remote(self, source_path, target_path):
        """
        本地和远程同时复制文件

        Args:
            source_path: 源文件路径（本地）
            target_path: 目标文件路径（本地，同时也会作为远程对象名称的基础）
        """
        try:
            source_path = Path(source_path)
            target_path = Path(target_path)

            # 1. 本地复制
            if not source_path.exists():
                minio_logger.error(f"源文件不存在: {source_path}")
                return False

            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 执行本地复制
            shutil.copy2(source_path, target_path)
            minio_logger.info(f"本地复制完成: {source_path} -> {target_path}")

            # 2. 远程复制（使用服务器端复制）
            source_object_name = self._get_relative_object_name(source_path)
            target_object_name = self._get_relative_object_name(target_path)

            # 检查源文件是否存在于远程
            try:
                self.client.stat_object(self.bucket_name, source_object_name)

                # 在服务器端直接复制对象
                self.client.copy_object(
                    self.bucket_name,
                    target_object_name,
                    CopySource(self.bucket_name,source_object_name)
                )
                minio_logger.info(f"远程复制完成（服务器端）: {source_object_name} -> {target_object_name}")

            except S3Error:
                # 如果远程源文件不存在，直接上传目标文件
                self.client.fput_object(
                    self.bucket_name,
                    target_object_name,
                    str(target_path)
                )
                minio_logger.info(f"远程文件已上传: {target_object_name}")

            return True

        except Exception as e:
            minio_logger.error(f"复制文件时出错: {e}")
            return False

    def move_file_local_remote(self, source_path, target_path):
        """
        本地和远程同时移动文件（包括重命名操作）

        Args:
            source_path: 源文件路径（本地）
            target_path: 目标文件路径（本地，同时也会作为远程对象名称的基础）
        """
        try:
            source_path = Path(source_path)
            target_path = Path(target_path)

            if not source_path.exists():
                minio_logger.error(f"源文件不存在: {source_path}")
                return False

            # 获取源文件和目标文件的远程对象名称
            source_object_name = self._get_relative_object_name(source_path)
            target_object_name = self._get_relative_object_name(target_path)

            # 1. 检查远程源文件是否存在
            try:
                self.client.stat_object(self.bucket_name, source_object_name)
                remote_exists = True
            except S3Error:
                remote_exists = False

            # 2. 本地移动
            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 执行本地移动
            shutil.move(str(source_path), str(target_path))
            minio_logger.info(f"本地移动完成: {source_path} -> {target_path}")

            # 3. 远程操作
            if remote_exists:
                if source_object_name != target_object_name:
                    # 如果对象名称不同，执行服务器端复制+删除
                    # 在服务器端直接复制对象
                    self.client.copy_object(
                        self.bucket_name,
                        target_object_name,
                        CopySource(self.bucket_name,source_object_name)
                    )
                    minio_logger.info(f"远程复制完成（服务器端）: {source_object_name} -> {target_object_name}")

                    # 删除源对象
                    self.client.remove_object(self.bucket_name, source_object_name)
                    minio_logger.info(f"远程删除源文件: {source_object_name}")
                else:
                    # 如果对象名称相同，只需要重新上传（因为文件内容可能已改变）
                    self.client.fput_object(
                        self.bucket_name,
                        target_object_name,
                        str(target_path)
                    )
                    minio_logger.info(f"远程文件已更新: {target_object_name}")
            else:
                # 如果远程源文件不存在，直接上传目标文件
                self.client.fput_object(
                    self.bucket_name,
                    target_object_name,
                    str(target_path)
                )
                minio_logger.info(f"远程文件已上传: {target_object_name}")

            return True

        except Exception as e:
            minio_logger.error(f"移动文件时出错: {e}")
            return False

    def copy_remote(self, source_object_name, target_object_name):
        """
        在远程直接复制文件（服务器端操作，不消耗客户端流量）

        Args:
            source_object_name: 源对象名称
            target_object_name: 目标对象名称
        """
        try:
            # 检查源对象是否存在
            self.client.stat_object(self.bucket_name, source_object_name)

            # 在服务器端直接复制对象
            self.client.copy_object(
                self.bucket_name,
                target_object_name,
                CopySource(self.bucket_name,source_object_name)
            )

            minio_logger.info(f"远程复制完成（服务器端）: {source_object_name} -> {target_object_name}")
            return True

        except S3Error as e:
            minio_logger.error(f"远程复制时出错: {e}")
            return False

    def move_remote(self, source_object_name, target_object_name):
        """
        在远程直接移动/重命名文件（服务器端操作，不消耗客户端流量）

        Args:
            source_object_name: 源对象名称
            target_object_name: 目标对象名称
        """
        try:
            # 检查源对象是否存在
            self.client.stat_object(self.bucket_name, source_object_name)

            # 1. 在服务器端直接复制对象
            self.client.copy_object(
                self.bucket_name,
                target_object_name,
                CopySource(self.bucket_name,source_object_name)
            )
            minio_logger.info(f"远程复制完成（服务器端）: {source_object_name} -> {target_object_name}")

            # 2. 删除源对象
            self.client.remove_object(self.bucket_name, source_object_name)
            minio_logger.info(f"远程删除源文件: {source_object_name}")

            return True

        except S3Error as e:
            minio_logger.error(f"远程移动时出错: {e}")
            return False

    def copy_both(self, source_path, target_path):
        """
        本地和远程同时复制文件，远程使用服务器端复制

        Args:
            source_path: 源文件路径（本地）
            target_path: 目标文件路径（本地）
        """
        try:
            source_path = Path(source_path)
            target_path = Path(target_path)

            # 1. 本地复制
            if not source_path.exists():
                minio_logger.error(f"源文件不存在: {source_path}")
                return False

            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 执行本地复制
            shutil.copy2(source_path, target_path)
            minio_logger.info(f"本地复制完成: {source_path} -> {target_path}")

            # 2. 远程复制（服务器端）
            source_object_name = self._get_relative_object_name(source_path)
            target_object_name = self._get_relative_object_name(target_path)

            return self.copy_remote(source_object_name, target_object_name)

        except Exception as e:
            minio_logger.error(f"同时复制文件时出错: {e}")
            return False

    def move_both(self, source_path, target_path):
        """
        本地和远程同时移动文件，远程使用服务器端移动

        Args:
            source_path: 源文件路径（本地）
            target_path: 目标文件路径（本地）
        """
        try:
            source_path = Path(source_path)
            target_path = Path(target_path)

            if not source_path.exists():
                minio_logger.error(f"源文件不存在: {source_path}")
                return False

            # 获取源文件和目标文件的远程对象名称
            source_object_name = self._get_relative_object_name(source_path)
            target_object_name = self._get_relative_object_name(target_path)

            # 1. 检查远程源文件是否存在
            try:
                self.client.stat_object(self.bucket_name, source_object_name)
                remote_exists = True
            except S3Error:
                remote_exists = False

            # 2. 本地移动
            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 执行本地移动
            shutil.move(str(source_path), str(target_path))
            minio_logger.info(f"本地移动完成: {source_path} -> {target_path}")

            # 3. 远程移动（服务器端）
            if remote_exists:
                return self.move_remote(source_object_name, target_object_name)
            else:
                # 如果远程源文件不存在，直接上传目标文件
                return self.upload_file(target_path)

        except Exception as e:
            minio_logger.error(f"同时移动文件时出错: {e}")
            return False

    def upload_directory(self, directory_path=None):
        """
        上传整个目录到MinIO

        Args:
            directory_path: 要上传的目录路径，默认为checkpoints目录
        """
        if directory_path is None:
            directory_path = self.local_dir

        try:
            directory_path = Path(directory_path).resolve()
            local_dir_path = Path(self.local_dir).resolve()

            if not directory_path.exists():
                minio_logger.error(f"目录不存在: {directory_path}")
                return False

            uploaded_count = 0
            for file_path in directory_path.rglob('*'):
                if file_path.is_file():
                    # 计算相对路径
                    if str(directory_path) == str(local_dir_path):
                        # 如果上传的就是checkpoints目录本身
                        rel_path = file_path.relative_to(local_dir_path)
                    else:
                        # 如果上传的是其他目录，在bucket中创建对应的子目录
                        rel_path = file_path.relative_to(directory_path.parent)

                    object_name = str(rel_path).replace('\\', '/')

                    # 上传文件
                    self.client.fput_object(
                        self.bucket_name,
                        object_name,
                        str(file_path)
                    )
                    uploaded_count += 1
                    minio_logger.info(f"上传文件: {object_name}")

            minio_logger.info(f"目录上传完成! 共上传 {uploaded_count} 个文件")
            return True

        except S3Error as e:
            minio_logger.error(f"上传目录时出错: {e}")
            return False

    def list_bucket_files(self):
        """列出bucket中的所有文件"""
        try:
            objects = self.client.list_objects(self.bucket_name, recursive=True)
            files = []
            for obj in objects:
                files.append(obj.object_name)
            return files
        except S3Error as e:
            minio_logger.error(f"列出文件时出错: {e}")
            return []

    def list_local_files(self):
        """列出本地checkpoints目录中的所有文件"""
        local_files = []
        for root, dirs, files in os.walk(self.local_dir):
            for file in files:
                full_path = Path(root) / file
                # 计算相对于checkpoints目录的路径
                rel_path = full_path.relative_to(self.local_dir)
                local_files.append(str(rel_path).replace('\\', '/'))
        return local_files


def main():
    minio_helper = MinIOHelper(
            endpoint='47.95.21.85:9000',
            access_key='admin',
            secret_key='admin666',
            secure=False
    )
    minio_helper.download_bucket()


if __name__ == "__main__":
    main()
