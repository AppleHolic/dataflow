from base import NetworkImages

import os
import ujson as json
import requests

import logging
logger = logging.getLogger(__name__)


class ILSVRC12(NetworkImages):
    def __init__(self, service_code, train_or_valid, shuffle=True):
        super(ILSVRC12, self).__init__(shuffle)

        base_path = 'http://twg.kakaocdn.net/%s/imagenet/ILSVRC/2012/object_localization/ILSVRC/' % service_code
        data_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../datas/ILSVRC/classification')
        temp_map = json.load(open(data_path+'/imagenet1000_classid_to_text_synsetid.json'))
        self.maps = {
            'idx2synset': {int(key): value['id'] for key, value in iter(temp_map.items())},
            'synset2idx': {value['id']: int(key) for key, value in iter(temp_map.items())},
            'idx2text':   {int(key): value['text'] for key, value in iter(temp_map.items())}
        }
        if train_or_valid in ['train', 'training']:
            _ = [
                line.decode('utf-8').strip().split(' ')[0]
                for line in requests.get(base_path + 'ImageSets/CLS-LOC/train_cls.txt').content.splitlines()
                if line.strip()
            ]
            self.datapoints = [
                [base_path + 'Data/CLS-LOC/train/'+line+'.JPEG', int(self.maps['synset2idx'][line.split('/')[0]])]
                for line in _
            ]
        elif train_or_valid in ['valid', 'validation']:
            synsets = [
                line.strip()
                for line in open(data_path+'/imagenet_2012_validation_synset_labels.txt').readlines()
            ]
            self.datapoints = [
                [base_path + 'Data/CLS-LOC/val/ILSVRC2012_val_%08d.JPEG' % (i+1), int(self.maps['synset2idx'][synset])]
                for i, synset in enumerate(synsets)
            ]
        else:
            raise ValueError('train_or_valid=%s is invalid argument must be a set train or valid' % train_or_valid)

def __test_one(args):
    logging.info('is_gevent : %s, nr_proc : %d, num_threads : %d' % (args.is_gevent, args.nr_proc, args.num_threads))
    ds = ILSVRC12(args.service_code, args.name, shuffle=False)
    if args.is_gevent:
        ds = ds.parallel_gevent(num_threads=args.num_threads)
    else:
        ds = ds.parallel(num_threads=args.num_threads)
    if args.name in ['train', 'training']:
        ds = df.PrefetchDataZMQ(ds, nr_proc=args.nr_proc)
    
    df.TestDataSpeed(ds, size=5000).start()

if __name__ == '__main__':
    import argparse
    import tensorpack.dataflow as df

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Imagenet Dataset on Kakao Example')
    parser.add_argument('--service-code', type=str, required=True,
                        help='licence key')
    parser.add_argument('--name', type=str, default='train',
                        help='train or valid')
    parser.add_argument('--is_gevent', type=bool, default=False, help='select gevent threaded parallel')
    parser.add_argument('--nr_proc', type=int, default=2)
    parser.add_argument('--num_threads', type=int, default=32)
    parser.add_argument('--test_all', type=bool, default=False)

    args = parser.parse_args()

    if args.test_all:
        test_list = [1, 2, 4, 8, 16, 32]
        for nr_proc in test_list:
            for num_threads in test_list:
                args.nr_proc = nr_proc
                args.num_threads = num_threads
                __test_one(args)
    else:
        __test_one(args)
