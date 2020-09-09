import argparse
import os
from utils import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='project_DPFIP',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='-1',
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str,
                                 default='./checkpoints',
                                 help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--activation', type=str, default='elu',
                                 help='relu or elu activation')
        self.parser.add_argument('--verbose', action='store_true',
                                 default=False, help='toggles verbose')
        self.parser.add_argument('--dataroot', type=str,
                                 default='../datasets')
        self.parser.add_argument("--batch_size", type=int, default=32)
        self.parser.add_argument("--val_split", type=int, default=10000)
        self.parser.add_argument("--test_split", type=int, default=20000)

        self.parser.add_argument("--num_workers", type=int, default=16)
        self.parser.add_argument('--how_many', type=int, default=50,
                                 help='how many test images to run')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')

        self.parser.add_argument('--input_nc', type=int, default=3,
                                 help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3,
                                 help='# of output image channels')
        self.parser.add_argument("--z_len", type=int, default=1024)

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,
                                 help='display window size')
        self.parser.add_argument('--tf_log', action='store_true',
                                 help='if specified, use tensorboard logging. Requires tensorflow installed')
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        self.opt.semantic_nc = self.opt.label_nc + \
                          (0 if self.opt.no_contain_dontcare_label else 1)
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        self.opt.classes_of_interest_ids = []
        if self.opt.use_bbox is not False:
            for str_class_id in self.opt.classes_of_interest.split(','):
                id = int(str_class_id)
                if id >= 0:
                    self.opt.classes_of_interest_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)
        if self.opt.local_rank == 0:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to the disk
        folder_name = "DPFIP"
        if self.opt.phase == "test":
            expr_dir = os.path.join(self.opt.checkpoints_dir, folder_name, self.opt.name, "test")
        else:
            expr_dir = os.path.join(self.opt.checkpoints_dir, folder_name,
                                    self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
