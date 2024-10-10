import os
ROOT = '/your/path/to/datasets/'
class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'webvision':
            return os.path.join(ROOT, 'webvision/')
        elif dataset == 'miniimagenet_preset': #CNWL
            return os.path.join(ROOT, 'mini-imagenet/')
        elif dataset == 'web-bird':
            return os.path.join(ROOT, 'web-bird/')
        elif dataset == 'web-car':
            return os.path.join(ROOT, 'web-car/')
        elif dataset == 'web-aircraft':
            return os.path.join(ROOT, 'web-aircraft/')
        else:
            raise NotImplementedError('Dataset {} not available.'.format(dataset))
        
