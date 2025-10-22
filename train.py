# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch, TextVidsModel
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # get label string
    data_label_str = train_dataset.get_label_dict()

    """FOR TEST SET"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    det_eval, output_file = None, None
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        val_dataset.json_file,
        val_dataset.split[0],
        tiou_thresholds = val_db_vars['tiou_thresholds'],
        ckpt_folder = ckpt_folder,
    )
    """FOR TEST SET"""

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    ### START: text model
    model_txt = TextVidsModel()
    model_txt = nn.DataParallel(model_txt, device_ids=cfg['devices'])
    optimizer_txt = make_optimizer(model_txt, cfg['opt'])
    scheduler_txt = make_scheduler(optimizer_txt, cfg['opt'], num_iters_per_epoch)
    ### END: text model


    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            # txt model
            model_txt.load_state_dict(checkpoint['state_dict_txt'])
            optimizer_txt.load_state_dict(checkpoint['optimizer_txt'])
            scheduler_txt.load_state_dict(checkpoint['scheduler_txt'])

            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    max_mAP = 0
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            model_txt,
            optimizer_txt,
            scheduler_txt,
            epoch,
            data_label_str,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq,
            combine_mode=args.combine_mode
        )

        mAP = valid_one_epoch(
            val_loader,
            model,
            model_txt,
            -1,
            data_label_str,
            evaluator=det_eval,
            output_file=output_file,
            ext_score_file=cfg['test_cfg']['ext_score_file'],
            tb_writer=None,
            print_freq=args.print_freq,
            combine_mode=args.combine_mode
        )

        # mAP_ema = valid_one_epoch(
        #     val_loader,
        #     model_ema.module,
        #     model_txt,
        #     -1,
        #     data_label_str,
        #     evaluator=det_eval,
        #     output_file=output_file,
        #     ext_score_file=cfg['test_cfg']['ext_score_file'],
        #     tb_writer=None,
        #     print_freq=args.print_freq
        # )

        # save mAP and epoch to log.txt
        with open(os.path.join(ckpt_folder, 'log.txt'), 'a') as fid:
            fid.write('Epoch: {:d}, mAP: {:.4f}\n'.format(epoch, mAP))
            # fid.write('Epoch: {:d}, mAP_ema: {:.4f}\n'.format(epoch, mAP_ema))

        if mAP > max_mAP:
            max_mAP = mAP
            print("New best mAP: {:.4f}".format(max_mAP))

            if args.save_checkpoint:
                save_states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'state_dict_txt': model_txt.state_dict(),
                    'scheduler_txt': scheduler_txt.state_dict(),
                    'optimizer_txt': optimizer_txt.state_dict(),
                }
                save_states['state_dict_ema'] = model_ema.module.state_dict()
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name='best_model.pth.tar'
                )

        # save max mAP to log.txt
        with open(os.path.join(ckpt_folder, 'log.txt'), 'a') as fid:
            fid.write('Best mAP: {:.4f}\n'.format(max_mAP))

    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--save-checkpoint', default=False, type=bool, help='save checkpoint or not')
    parser.add_argument('--combine-mode', default=0, type=int, help='combine mode')

    args = parser.parse_args()
    main(args)
