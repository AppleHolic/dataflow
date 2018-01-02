import time

import requests
import numpy as np
import cv2
import gevent
from gevent import pool

import tensorpack.dataflow as df
from tensorpack.dataflow.base import RNGDataFlow

import logging
logger = logging.getLogger(__name__)


class NetworkImages(RNGDataFlow):
    def __init__(self, shuffle=False):
        # will be implement datapoints
        # self.datapoints = [['http://image_url', labels...], ]
        self.datapoints = []
        self.shuffle = shuffle
        self.is_parallel = False
        
    def size(self):
        return len(self.datapoints)

    @staticmethod
    def request(url, max_trials=5):
        for trial in range(max_trials):
            try:
                resp = requests.get(url)
                if resp.status_code // 100 != 2 or len(resp.content) == 0:
                    logger.warning('request failed http status code=%d url=%s' % (resp.status_code, url))
                    time.sleep(0.05)
                    continue
                return resp.content
            except requests.ConnectionError as ce:
                logger.warning('Connection Aborted : %s, url : %s' % (str(ce), url))
                time.sleep(0.5)
            except Exception as e:
                logger.warning('request failed error=%s url=%s' % (str(e), url))
        return None

    @staticmethod
    def map_func_download(datapoint):
        content = NetworkImages.request(datapoint[0])
        return [content] + datapoint[1:]

    @staticmethod
    def map_func_decode(datapoint):        
        img = cv2.imdecode(np.fromstring(datapoint[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        return [img] + datapoint[1:]

    @staticmethod
    def map_merged_func(datapoint):
        datapoint = NetworkImages.map_func_download(datapoint)
        img = cv2.imdecode(np.fromstring(datapoint[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        return [img] + datapoint[1:]

    @staticmethod
    def map_gevent(pool, datapoints, func=map_func_download):
        jobs = []
        for dp in datapoints:
            jobs.append(pool.spawn(func, dp))
        gevent.joinall(jobs)
        return map(lambda x: x.value, jobs)

    def get_data(self):
        idxs = np.arange(len(self.datapoints))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for i, k in enumerate(idxs):
            dp = self.datapoints[k]
            if not self.is_parallel:
                dp = NetworkImages.map_func_decode(NetworkImages.map_func_download(dp))
            yield dp

    def _gevent_get_data(self):
        idxs = np.arange(len(self.datapoints))
        if self.shuffle:
            self.rng.shuffle(idxs)

        jobs = []
        _pool = pool.Pool(self.num_threads)
        for i in idxs:
            dp = self.datapoints[i]
            jobs.append(_pool.spawn(NetworkImages.map_func_download, dp))
            if len(jobs) == self.num_threads:
                gevent.joinall(jobs)
                for job in jobs:
                    yield job.value
                del jobs[:]

    def partitioning(self, num_partition, partition_index=0):
        if num_partition <= partition_index:
            raise ValueError('partition_index(=%d) is must be a smaller than num_partition(=%d)' % (
                partition_index, num_partition))
        self.datapoints = self.datapoints[partition_index::num_partition]
        return self

    def parallel(self, num_threads, buffer_size=200, strict=False):
        self.is_parallel = True
        ds = self
        ds = df.MultiThreadMapData(ds,
                                   nr_thread=num_threads, map_func=NetworkImages.map_func_download,
                                   buffer_size=buffer_size, strict=strict)
        # ds = df.PrefetchDataZMQ(ds, nr_proc=1)  # to reduce GIL contention.
        ds = df.MapData(ds, func=NetworkImages.map_func_decode)
        return ds

    def parallel_gevent(self, num_threads):
        self.is_parallel = True
        self.num_threads = num_threads
        self.get_data = self._gevent_get_data
        ds = df.MapData(self, func=NetworkImages.map_func_decode)
        return ds
