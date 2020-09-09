import os
from torch.utils import data
from tqdm import tqdm
import torchvision.transforms as transform
import torch
from options.train_options import TrainOptions
import numpy as np
import math
import time
import utils.util as util
from utils import html
from utils.visualizer import Visualizer
from collections import OrderedDict
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from model import VaeGanModule
def lcm(a, b): return abs(a * b) / math.gcd(a, b) if a and b else 0


# dataset
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.backends.cudnn.benchmark = True

device = \
    [torch.device(f"cuda:0" if use_cuda else "cpu")]

def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


def train(gpu, opt):
    device = torch.device(f"cuda:{opt.gpu_ids[0]}" if use_cuda else "cpu")
    opt.print_freq = lcm(opt.print_freq, opt.batch_size)

    folder_name = "DPFIP"

    iter_path = os.path.join(opt.checkpoints_dir, folder_name, opt.name,
                             'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',',
                                                 dtype=int)
        except FileNotFoundError as e:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (
        start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0
    dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.CenterCrop(178),
                                       transforms.Resize(128),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5)),
                                   ]))
    # Create the dataloader
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.num_workers)

    dataset_size = len(trainloader) * opt.batch_size
    visualizer = Visualizer(opt)
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq
    model = VaeGanModule(opt, device)

    if use_cuda:
        model = model.cuda()
    optimizer_vae, optimizer_D= model.optimizer_vae, model.optimizer_D

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        for i, data in enumerate(iter(trainloader)):
            compute_g_loss = True
            images = data[0].to(device)
            if total_steps % opt.print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            save_fake = total_steps % opt.display_freq == display_delta
            losses, fake_images = model(images)
            output_images = fake_images.detach()
            # sum per device losses
            losses = [
                torch.mean(x) if not isinstance(x,
                                                int) and x is not None else x
                for x in losses]
            if use_cuda:
                loss_dict = dict(zip(model.loss_names, losses))
            else:
                loss_dict = dict(zip(model.loss_names, losses))
            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            ############### Backward Pass ####################
            # update generator weights
            # Update G
            loss_G = loss_dict['G_GAN'] + \
                         loss_dict.get("G_Image_Rec", 0) + \
                         loss_dict.get("G_KL_image", 0)
            optimizer_vae.zero_grad()
            loss_G.backward()
            optimizer_vae.step()
            optimizer_vae.zero_grad()

            # update discriminator weights
            loss_D.backward()
            optimizer_D.step()
            optimizer_D.zero_grad()

            ############## Display results and errors ##########
            ### print out errors

            if total_steps % opt.print_freq == print_delta:
                errors = {k: v.data.item() if not isinstance(v,
                                                             int) and v is not None else v
                          for k, v
                          in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errors,
                                                t)
                visualizer.plot_current_errors(errors, total_steps)
                # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

            ### display output images
            visuals = OrderedDict([('real_image',
                                        util.tensor2im(images[0])),
                                       ('fake_image',
                                        util.tensor2im(output_images[0]))])
            visualizer.display_current_results(visuals, epoch,
                                                   total_steps)
            ### save latest model
            if (total_steps % opt.save_latest_freq == save_delta):

                print(
                    'saving the latest model (epoch %d, total_steps %d)' % (
                        epoch, total_steps))
                if use_cuda:
                    model.save('latest')
                else:
                    model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',',
                           fmt='%d')
            if epoch_iter >= dataset_size:
                break
            # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay,
                   time.time() - epoch_start_time))

        ### save model for this epoch
        if (epoch % opt.save_epoch_freq == 0):
            print('saving the model at the end of epoch %d, iters %d' % (
                epoch, total_steps))
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.update_learning_rate()



if __name__ == "__main__":
    opt = TrainOptions().parse()
    train(None, opt)
