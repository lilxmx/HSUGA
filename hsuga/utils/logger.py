import logging
import os
import time
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """Logging manager with TensorBoard support."""

    def __init__(self, args):
        self.args = args
        self._create_logger()

    def _create_logger(self):
        dataset = getattr(self.args, 'dataset', 'unknown')
        model_name = getattr(self.args, 'model_name', 'model') if hasattr(self.args, 'model_name') \
            else getattr(getattr(self.args, 'model', None), 'name', 'model')
        
        main_path = os.path.join('./log/', dataset, model_name, '')
        os.makedirs(main_path, exist_ok=True)

        now_str = time.strftime("%m%d%H%M%S", time.localtime())
        log_enabled = getattr(self.args, 'log', True)

        if log_enabled:
            os.makedirs(main_path + now_str + '/', exist_ok=True)
            folder_name = main_path + now_str + '/tensorboard/'
            batch_size = getattr(self.args, 'train_batch_size', 
                                 getattr(getattr(self.args, 'training', None), 'batch_size', 128))
            lr = getattr(self.args, 'lr',
                        getattr(getattr(self.args, 'training', None), 'lr', 0.001))
            file_path = now_str + f'/bs{batch_size}_lr{lr}.txt'
        else:
            folder_name = main_path + '/default/'
            file_path = 'default/log.txt'

        self.writer = SummaryWriter(folder_name)

        self.logger = logging.getLogger(model_name + '_' + now_str)
        self.logger.setLevel(logging.DEBUG)

        log_path = main_path + file_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        self.fh.setLevel(logging.DEBUG)
        fm = logging.Formatter("%(asctime)s-%(message)s")
        self.fh.setFormatter(fm)
        self.logger.addHandler(self.fh)

        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.logger.addHandler(self.ch)

        self.now_str = now_str

    def end_log(self):
        self.logger.removeHandler(self.fh)
        self.logger.removeHandler(self.ch)

    def get_logger(self):
        return self.logger, self.writer

    def get_now_str(self):
        return self.now_str
