#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import os
import re
import time
from qcloud_cos import CosConfig, CosS3Client
from qcloud_cos.cos_exception import CosClientError, CosServiceError
from rag.utils import singleton
from rag import settings


@singleton
class RAGFlowCOS:
    def __init__(self):
        self.conn = None
        self.cos_config = settings.COS
        # 支持从环境变量读取配置（优先级：环境变量 > 配置文件）
        self.secret_id = (
            os.environ.get('COS_SECRET_ID') or
            self.cos_config.get('secret_id', None)
        )
        self.secret_key = (
            os.environ.get('COS_SECRET_KEY') or
            self.cos_config.get('secret_key', None)
        )
        self.region = (
            os.environ.get('COS_REGION') or
            self.cos_config.get('region', None)
        )
        self.scheme = (
            os.environ.get('COS_SCHEME') or
            self.cos_config.get('scheme', 'https')
        )
        self.appid = (
            os.environ.get('COS_APPID') or
            self.cos_config.get('appid', None)
        )
        self.bucket = (
            os.environ.get('COS_BUCKET') or
            self.cos_config.get('bucket', None)
        )
        self.prefix_path = (
            os.environ.get('COS_PREFIX_PATH') or
            self.cos_config.get('prefix_path', None)
        )
        self.__open__()
    
    def _normalize_bucket_name(self, bucket):
        """
        规范化bucket名称，确保格式为 bucketname-appid
        腾讯云COS要求所有bucket名称必须遵循 bucketname-appid 格式
        
        如果bucket名称已经包含appid，直接返回；否则自动添加appid
        """
        if not bucket or not self.appid:
            return bucket
        
        # 如果已经包含appid（格式为 bucketname-appid），直接返回
        if bucket.endswith(f"-{self.appid}"):
            return bucket
        
        # 检查是否已经包含其他appid格式（最后一部分是数字）
        # 如果已经包含appid格式，使用配置的appid替换
        parts = bucket.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            # 已经包含appid格式，但可能不是当前配置的appid
            # 使用配置的appid替换
            bucket = f"{parts[0]}-{self.appid}"
        else:
            # 不包含appid，添加appid
            bucket = f"{bucket}-{self.appid}"
        
        return bucket

    @staticmethod
    def use_default_bucket(method):
        def wrapper(self, bucket, *args, **kwargs):
            # If there is a default bucket, use the default bucket
            actual_bucket = self.bucket if self.bucket else bucket
            # 规范化bucket名称（添加appid）
            actual_bucket = self._normalize_bucket_name(actual_bucket)
            return method(self, actual_bucket, *args, **kwargs)
        return wrapper
    
    @staticmethod
    def use_prefix_path(method):
        def wrapper(self, bucket, fnm, *args, **kwargs):
            # If the prefix path is set, use the prefix path
            fnm = f"{self.prefix_path}/{fnm}" if self.prefix_path else fnm
            return method(self, bucket, fnm, *args, **kwargs)
        return wrapper

    def __open__(self):
        try:
            if self.conn:
                self.__close__()
        except Exception:
            pass

        try:
            config = CosConfig(
                Region=self.region,
                SecretId=self.secret_id,
                SecretKey=self.secret_key,
                Scheme=self.scheme
            )
            self.conn = CosS3Client(config)
            normalized_bucket = self._normalize_bucket_name(self.bucket) if self.bucket else None
            logging.info(f"[COS] 腾讯云COS对象存储已启用 - Region: {self.region}, Scheme: {self.scheme}, AppId: {self.appid or '未设置'}, DefaultBucket: {normalized_bucket or '未设置'}, PrefixPath: {self.prefix_path or '无'}")
        except Exception:
            logging.exception(f"Fail to connect to COS at region {self.region}")

    def __close__(self):
        del self.conn
        self.conn = None

    @use_default_bucket
    def bucket_exists(self, bucket):
        try:
            logging.debug(f"head_bucket bucketname {bucket}")
            self.conn.head_bucket(Bucket=bucket)
            exists = True
        except (CosClientError, CosServiceError) as e:
            # bucket不存在时返回False，不记录错误日志
            if isinstance(e, CosServiceError):
                error_code = e.get_error_code()
                if error_code in ['NoSuchBucket', 'NoSuchResource']:
                    exists = False
                else:
                    logging.warning(f"head_bucket error {bucket}: {error_code} - {e.get_error_msg()}")
                    exists = False
            else:
                logging.warning(f"head_bucket error {bucket}: {str(e)}")
                exists = False
        except Exception as e:
            logging.warning(f"head_bucket error {bucket}: {str(e)}")
            exists = False
        return exists

    def health(self, bucket=None):
        """
        健康检查：测试COS连接和基本操作
        
        Args:
            bucket: 可选，指定测试用的bucket。如果不指定，使用配置的默认bucket
        
        Returns:
            bool: 健康检查是否通过
        """
        # 如果没有指定bucket，使用配置的默认bucket
        if bucket is None:
            bucket = self.bucket
            if not bucket:
                logging.warning("health check: no bucket specified and no default bucket configured")
                return False
        
        # 规范化bucket名称
        bucket = self._normalize_bucket_name(bucket)
        
        fnm = "txtxtxtxt1"
        fnm, binary = f"{self.prefix_path}/{fnm}" if self.prefix_path else fnm, b"_t@@@1"
        if not self.bucket_exists(bucket):
            try:
                self.conn.create_bucket(Bucket=bucket)
                logging.debug(f"create bucket {bucket} ********")
            except Exception:
                logging.exception(f"Fail to create bucket {bucket}")

        try:
            self.conn.put_object(
                Bucket=bucket,
                Body=binary,
                Key=fnm
            )
            return True
        except Exception:
            logging.exception(f"Fail to put object in health check")
            return False

    def get_properties(self, bucket, key):
        return {}

    def list(self, bucket, dir, recursive=True):
        return []

    @use_prefix_path
    @use_default_bucket
    def put(self, bucket, fnm, binary):
        logging.debug(f"bucket name {bucket}; filename :{fnm}:")
        max_retries = 3
        for retry in range(max_retries):
            try:
                if not self.bucket_exists(bucket):
                    try:
                        self.conn.create_bucket(Bucket=bucket)
                        logging.info(f"create bucket {bucket} ********")
                        # COS创建bucket后可能需要等待一段时间才能使用（最终一致性）
                        # 等待并重试bucket_exists确认bucket已创建
                        for wait_retry in range(5):
                            time.sleep(0.5)  # 等待0.5秒
                            if self.bucket_exists(bucket):
                                break
                            if wait_retry == 4:
                                logging.warning(f"Bucket {bucket} created but not immediately available, will retry put operation")
                    except CosServiceError as e:
                        # 如果bucket已存在，忽略错误
                        if e.get_error_code() not in ['BucketAlreadyExists', 'BucketAlreadyOwnedByYou']:
                            logging.exception(f"Fail to create bucket {bucket}")
                    except Exception:
                        logging.exception(f"Fail to create bucket {bucket}")
                
                self.conn.put_object(
                    Bucket=bucket,
                    Body=binary,
                    Key=fnm
                )
                return True
            except CosServiceError as e:
                error_code = e.get_error_code()
                if error_code == 'NoSuchBucket' and retry < max_retries - 1:
                    # bucket不存在，可能是刚创建还未生效，等待后重试
                    logging.warning(f"Bucket {bucket} not available yet, retrying... (attempt {retry + 1}/{max_retries})")
                    time.sleep(1)
                    continue
                else:
                    logging.exception(f"Fail put {bucket}/{fnm}: {error_code} - {e.get_error_msg()}")
                    if retry < max_retries - 1:
                        time.sleep(1)
                        continue
                    return False
            except Exception as e:
                logging.exception(f"Fail put {bucket}/{fnm}")
                if retry < max_retries - 1:
                    self.__open__()
                    time.sleep(1)
                    continue
                return False
        return False

    @use_prefix_path
    @use_default_bucket
    def rm(self, bucket, fnm):
        try:
            self.conn.delete_object(Bucket=bucket, Key=fnm)
        except CosServiceError as e:
            # 如果资源不存在，认为删除成功（幂等性）
            if e.get_error_code() in ['NoSuchResource', 'NoSuchKey', '404']:
                logging.debug(f"Object {bucket}/{fnm} does not exist, consider deletion successful")
                return
            else:
                logging.exception(f"Fail rm {bucket}/{fnm}")
        except Exception:
            logging.exception(f"Fail rm {bucket}/{fnm}")

    @use_prefix_path
    @use_default_bucket
    def get(self, bucket, fnm):
        max_retries = 3
        for retry in range(max_retries):
            try:
                response = self.conn.get_object(Bucket=bucket, Key=fnm)
                object_data = response['Body'].read()
                return object_data
            except CosServiceError as e:
                error_code = e.get_error_code()
                if error_code == 'NoSuchBucket' and retry < max_retries - 1:
                    # bucket不存在，可能是刚创建还未生效，等待后重试
                    logging.warning(f"Bucket {bucket} not available yet, retrying... (attempt {retry + 1}/{max_retries})")
                    time.sleep(1)
                    continue
                else:
                    logging.exception(f"fail get {bucket}/{fnm}: {error_code} - {e.get_error_msg()}")
                    if retry < max_retries - 1:
                        time.sleep(1)
                        continue
                    return None
            except Exception as e:
                logging.exception(f"fail get {bucket}/{fnm}")
                if retry < max_retries - 1:
                    self.__open__()
                    time.sleep(1)
                    continue
                return None
        return None

    @use_prefix_path
    @use_default_bucket
    def obj_exist(self, bucket, fnm):
        try:
            self.conn.head_object(Bucket=bucket, Key=fnm)
            return True
        except CosServiceError as e:
            # 文件不存在时返回False，不抛出异常
            error_code = e.get_error_code()
            if error_code in ['NoSuchKey', 'NoSuchResource', '404']:
                return False
            else:
                # 其他错误记录日志但不抛出异常，返回False
                logging.warning(f"obj_exist error {bucket}/{fnm}: {error_code} - {e.get_error_msg()}")
                return False
        except Exception as e:
            # 捕获所有其他异常，记录日志但不抛出，返回False
            logging.exception(f"obj_exist error {bucket}/{fnm}")
            return False

    @use_prefix_path
    @use_default_bucket
    def get_presigned_url(self, bucket, fnm, expires):
        for _ in range(10):
            try:
                url = self.conn.get_presigned_download_url(
                    Bucket=bucket,
                    Key=fnm,
                    Expired=expires
                )
                return url
            except Exception:
                logging.exception(f"fail get url {bucket}/{fnm}")
                self.__open__()
                time.sleep(1)
        return
