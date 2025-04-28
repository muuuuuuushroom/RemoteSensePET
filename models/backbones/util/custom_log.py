import os
import datetime
import logging
import yaml

from .misc import is_main_process


def load_config(config_path):
    assert os.path.exists(config_path), f'cfg: {config_path} not exists'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def update_args_with_config(args, config):
    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if not hasattr(args, sub_key):
                    setattr(args, sub_key, sub_value)
        else:
            if not hasattr(args, key):
                setattr(args, key, value)
    return args


def save_config(args, save_path):
    config_dict = vars(args)
    with open(save_path, 'w') as file:
        yaml.dump(config_dict, file)


class printLog():
    def __init__(self, logger) -> None:
        self.logger = logger
    
    def __call__(self, msg, level='info'):
        if is_main_process():
            if level == 'info':
                self.logger.info(msg)
            elif level == 'warn':
                self.logger.warn(msg)
            elif level == 'error':
                self.logger.error(msg)
            else:
                raise NotImplementedError


def logger_init(args):
    '''
    logger init & write base config message
    
    input:
        args: terminal arguments
        save_path: 
    output:
        logger: logger hander for recording train & test message
    '''
    runtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    save_path = os.path.join("./outputs", args.dataset_file, args.output_dir)
    # args.exp = f'log/{args.exp}'
    rundir = os.path.join(save_path, runtime)
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    
    with open(f'{save_path}/{runtime}/{runtime}.log', 'w+') as f:
        for k, v in args.__dict__.items():
            f.write(f'{k} = {v}\n')
        f.write('\n')

    logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='## %Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    filename=save_path+f'/{runtime}/{runtime}.log',
                    filemode='a+')
    logger = logging.getLogger(runtime)
    logger.setLevel(level=logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    printlog = printLog(logger)

    return printlog
