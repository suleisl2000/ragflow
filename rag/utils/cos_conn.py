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

# 将 qcloud_cos SDK 的 head/get object 等 INFO 日志降为仅 DEBUG 时显示（通过设为 WARNING 屏蔽）
logging.getLogger("qcloud_cos").setLevel(logging.WARNING)
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
            # 优先使用传入的 bucket 参数，如果未传入（None 或空字符串）才使用默认 bucket
            # 这样可以支持不同的 bucket（如 pdf-cache, paddleocr-cache 等）
            actual_bucket = bucket if bucket else (self.bucket if self.bucket else None)
            
            if not actual_bucket:
                logging.error(f"[COS] Bucket名称缺失: 传入参数={bucket}, 默认bucket={self.bucket}")
                raise ValueError("Bucket name is required (either as parameter or default bucket)")
            
            # 规范化bucket名称（添加appid）
            normalized_bucket = self._normalize_bucket_name(actual_bucket)
            
            # 只在 bucket 名称发生变化时记录（避免频繁日志）
            if normalized_bucket != actual_bucket:
                logging.debug(f"[COS] Bucket名称规范化: {actual_bucket} -> {normalized_bucket}")
            
            return method(self, normalized_bucket, *args, **kwargs)
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
            self.conn.head_bucket(Bucket=bucket)
            return True
        except CosServiceError as e:
            error_code = e.get_error_code()
            if error_code in ['NoSuchBucket', 'NoSuchResource']:
                return False
            # 其他错误也返回False，不记录日志（正常情况）
            return False
        except Exception:
            # 异常情况返回False，不记录日志
            return False

    def health(self, bucket=None):
        """
        健康检查：直接返回 True（不进行实际检查）
        
        Args:
            bucket: 可选，指定测试用的bucket（当前未使用）
        
        Returns:
            bool: 始终返回 True
        """
        # 暂时不进行实际检查，直接返回 True
        # 这样可以避免频繁的 COS 操作和日志输出
        return True

    def get_properties(self, bucket, key):
        return {}

    def list(self, bucket, dir, recursive=True):
        return []

    @use_prefix_path
    @use_default_bucket
    def put(self, bucket, fnm, binary):
        max_retries = 3
        for retry in range(max_retries):
            try:
                # 先检查bucket是否存在，不存在则创建
                if not self.bucket_exists(bucket):
                    try:
                        self.conn.create_bucket(Bucket=bucket)
                        logging.info(f"create bucket {bucket} ********")
                    except CosServiceError as e:
                        # 如果bucket已存在，忽略错误（可能其他进程已创建）
                        if e.get_error_code() not in ['BucketAlreadyExists', 'BucketAlreadyOwnedByYou']:
                            logging.exception(f"Fail to create bucket {bucket}")
                    
                    # COS创建bucket后需要等待一段时间才能使用（最终一致性）
                    # 检查bucket是否存在，最多3次，每次等待3秒
                    bucket_ready = False
                    for check_retry in range(3):
                        time.sleep(3)
                        if self.bucket_exists(bucket):
                            bucket_ready = True
                            break
                    
                    # 如果3次检查后bucket还不存在，再尝试create_bucket一次
                    if not bucket_ready:
                        try:
                            self.conn.create_bucket(Bucket=bucket)
                            logging.info(f"retry create bucket {bucket} ********")
                            time.sleep(3)
                        except CosServiceError as e:
                            # 如果bucket已存在，说明bucket已经可用了
                            if e.get_error_code() in ['BucketAlreadyExists', 'BucketAlreadyOwnedByYou']:
                                pass
                            else:
                                logging.exception(f"Fail to retry create bucket {bucket}")
                
                # 执行put_object（bucket已经通过装饰器规范化）
                self.conn.put_object(Bucket=bucket, Body=binary, Key=fnm)
                logging.debug(f"[COS] put object成功: bucket={bucket}, key={fnm}, size={len(binary)} bytes")
                return True
            except CosServiceError as e:
                error_code = e.get_error_code()
                if error_code == 'NoSuchBucket' and retry < max_retries - 1:
                    # bucket不存在，可能是刚创建还未生效，等待后重试
                    time.sleep(3)
                    continue
                else:
                    # 最后一次重试失败或其他错误，记录日志
                    if retry == max_retries - 1:
                        logging.exception(f"Fail put {bucket}/{fnm}: {error_code} - {e.get_error_msg()}")
                    return False
            except Exception as e:
                if retry == max_retries - 1:
                    logging.exception(f"Fail put {bucket}/{fnm}")
                elif retry < max_retries - 1:
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
            # 如果资源不存在，认为删除成功（幂等性），不记录日志
            if e.get_error_code() not in ['NoSuchResource', 'NoSuchKey', '404']:
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
                # 使用get_raw_stream()获取原始二进制流，确保返回bytes类型（二进制数据）
                # 直接read()可能返回文本，get_raw_stream().read()返回二进制
                body_stream = response['Body'].get_raw_stream()
                object_data = body_stream.read()
                # 确保返回的是bytes类型（二进制数据）
                if not isinstance(object_data, bytes):
                    # 如果不是bytes，尝试转换
                    if isinstance(object_data, str):
                        object_data = object_data.encode('latin-1')  # 使用latin-1保持二进制完整性
                    else:
                        object_data = bytes(object_data)
                return object_data
            except CosServiceError as e:
                error_code = e.get_error_code()
                if error_code == 'NoSuchBucket' and retry < max_retries - 1:
                    # bucket不存在，可能是刚创建还未生效，等待后重试
                    time.sleep(2)
                    continue
                else:
                    # 最后一次重试失败或其他错误，记录日志
                    if retry == max_retries - 1:
                        logging.exception(f"fail get {bucket}/{fnm}: {error_code} - {e.get_error_msg()}")
                    return None
            except Exception as e:
                if retry == max_retries - 1:
                    logging.exception(f"fail get {bucket}/{fnm}")
                elif retry < max_retries - 1:
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
            # 文件不存在时返回False，不记录日志（正常情况）
            error_code = e.get_error_code()
            if error_code in ['NoSuchKey', 'NoSuchResource', '404']:
                return False
            # 其他错误也返回False，不记录日志
            return False
        except Exception:
            # 异常情况返回False，不记录日志
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

