import os
import yaml
import time
import torch
import argparse
import importlib
import torch.distributed
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from shutil import copy2
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
import numpy as np


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()
    rt /= world_size
    return rt


def get_args(ngpus_per_node):
    # command line args
    parser = argparse.ArgumentParser(
        description='Doppelgangers: Learning to Disambiguate Images of Similar Structures')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

    # distributed training
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Total number of batches (None will read batch size from the [cfg]).')
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')
    parser.add_argument('--sync_bn', action='store_true',
                        help="Whether use syncrhonized batch normalization")

    # Resume:
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")

    # Test run:
    parser.add_argument('--test_run', default=False, action='store_true')
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    #  Create log_name
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')

    # Currently save dir and log_dir are the same
    config.log_name = "logs/%s_%s" % (cfg_file_name, run_time)
    config.save_dir = "logs/%s_%s" % (cfg_file_name, run_time)
    config.log_dir = "logs/%s_%s" % (cfg_file_name, run_time)
    if args.local_rank % ngpus_per_node == 0:
        os.makedirs(config.log_dir+'/config', exist_ok=True)
        copy2(args.config, config.log_dir+'/config')
    return args, config


def main_worker(gpu, ngpus_per_node, cfg, args):
    # basic setup
    cudnn.benchmark = True
    args.gpu = gpu
    args.rank = gpu
    assert args.gpu is not None
    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    print("Use GPU: {} for training".format(args.gpu))
    print("Rank: %d\tNGPUs: %d\tGPU:%d" % (args.rank, ngpus_per_node, gpu))
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank % ngpus_per_node == 0:
        writer = SummaryWriter(logdir=cfg.log_name)
    else:
        writer = None

    with torch.cuda.device(args.gpu):
        trainer_lib = importlib.import_module(cfg.trainer.type)
        trainer = trainer_lib.Trainer(cfg, args)
        def wrapper(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[args.gpu], output_device=args.gpu,
                find_unused_parameters=True) # check_reduction=True, 

        trainer.multi_gpu_wrapper(wrapper)
        torch.cuda.set_device(args.gpu)
        if args.batch_size is not None:
            cfg.data.batch_size = int(args.batch_size / ngpus_per_node)
        cfg.workers = 0

        # initialize datasets and loaders
        data_lib = importlib.import_module(cfg.data.type)
        tr_dataset, te_dataset = data_lib.get_datasets(cfg.data)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            tr_dataset)

        train_loader = torch.utils.data.DataLoader(
            dataset=tr_dataset, batch_size=cfg.data.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=cfg.data.num_workers, pin_memory=True,
            sampler=train_sampler, drop_last=True, worker_init_fn=init_np_seed)
        test_loader = torch.utils.data.DataLoader(
            dataset=te_dataset, batch_size=cfg.data.test.batch_size, shuffle=False,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=False,
            worker_init_fn=init_np_seed)
        

        start_epoch = 0
        start_time = time.time()

        if args.resume:
            if args.pretrained is not None:
                start_epoch = trainer.resume(args.pretrained, multi_gpu=True) + 1
            else:
                start_epoch = trainer.resume(cfg.resume.dir, multi_gpu=True)

        # If test run, go through the validation loop first
        if args.test_run:
            if args.rank % ngpus_per_node == 0:
                trainer.save_multigpu(epoch=-1, step=-1)

            val_info = trainer.validate(test_loader, epoch=-1)
            trainer.log_val(val_info, writer=writer, epoch=-1)
            trainer.log_val(val_info, writer=writer, step=-1)

        # main training loop
        print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs))
        step = 0
        for epoch in range(start_epoch, cfg.trainer.epochs):
            train_loader.sampler.set_epoch(epoch)

            # train for one epoch
            for bidx, data in enumerate(train_loader):
                step = bidx + len(train_loader) * epoch + 1
                logs_info = trainer.update(data)
                if step % int(cfg.viz.log_freq) == 0:
                    duration = time.time() - start_time
                    start_time = time.time()
                    print("Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loss %2.5f"
                        % (epoch, bidx, len(train_loader), duration,
                            logs_info['loss']))                    
                    trainer.log_train(
                        logs_info, data,
                        writer=writer, epoch=epoch, step=step)
                if int(cfg.viz.val_freq)>0:
                    if step % int(cfg.viz.val_freq) == 0:
                        val_info = trainer.validate(test_loader, epoch=epoch)
                        trainer.log_val(val_info, writer=writer, step=step)

            if args.rank % ngpus_per_node == 0:
                # Save first so that even if the visualization bugged,
                # we still have something
                if (epoch + 1) % int(cfg.viz.save_freq) == 0:
                    trainer.save_multigpu(epoch=epoch, step=step)

            
            if (epoch + 1) % int(cfg.viz.save_freq) == 0:
                # and int(cfg.viz.val_freq) > 0:
                val_info = trainer.validate(test_loader, epoch=epoch)
                trainer.log_val(val_info, writer=writer, epoch=epoch)

            # Signal the trainer to cleanup now that an epoch has ended
            trainer.epoch_end(epoch, writer=writer)
        writer.close()


def main():
    # command line args
    ngpus_per_node = torch.cuda.device_count()
    args, cfg = get_args(ngpus_per_node)

    if args.gpu is not None:
        print('WARN: You have chosen a specific GPU. This will completely '
              'disable data parallelism.')
        assert False

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.sync_bn:
        assert False, "Do not support syncrhonized batch norm so far"

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    args.world_size = ngpus_per_node
    mp.spawn(main_worker, args=(ngpus_per_node, cfg, args), nprocs=ngpus_per_node)


if __name__ == '__main__':
    main()