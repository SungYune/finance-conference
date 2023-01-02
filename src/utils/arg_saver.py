import os
import matplotlib.pyplot as plt
from pathlib import Path


def arg_saver(args, version):
    # save args
    args_dict = vars(args)
    path_to_save = os.path.join(
        f'src/lightning_logs/{args.model_id}/{args.model_arch_desc}',
        f'version_{version}',
        'args.txt')

    with open(path_to_save, mode='w') as f:
        for key, value in args_dict.items():
            f.write(f'--{key} {value} \\ \n')
