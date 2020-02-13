# -*- coding: utf-8 -*-

# to avoid "_tkinter.TclError: no display name and no $DISPLAY environment variable" error
import matplotlib as mpl

mpl.use('Agg')

import argparse
import chainer
import h5py
import numpy as np
import os
import sys

import torch
import torch.nn.functional as F
import ignite
from ignite.contrib.handlers.param_scheduler import LRScheduler
import chainer_pytorch_migration as cpm
import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as extensions

import formulanet
import holstep
import symbols

sys.setrecursionlimit(10000)


def main():
    parser = argparse.ArgumentParser(description='chainer formulanet trainer')

    parser.add_argument('--chainermn', action='store_true', help='Use ChainerMN')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--devices', type=str, default='',
                        help='Comma-separated list of devices specifier.')
    parser.add_argument('--dataset', '-i', default="holstep",
                        help='Directory of holstep repository')
    parser.add_argument('--out', '-o', default='results',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    #    parser.add_argument('--seed', type=int, default=0,
    #                        help='Random seed')
    #    parser.add_argument('--snapshot_interval', type=int, default=10000,
    #                        help='Interval of snapshot')
    #    parser.add_argument('--display_interval', type=int, default=100,
    #                        help='Interval of displaying log to console')
    parser.add_argument('--conditional', action='store_true', help='Use contional model')
    parser.add_argument('--preserve-order', action='store_true', help='Use order-preserving model')
    parser.add_argument('--steps', type=int, default="3", help='Number of update steps')

    args = parser.parse_args()

    if args.chainermn:
        # matplotlib.font_manager should be imported before mpi4py.MPI
        # to avoid MPI issue with fork() system call.
        import matplotlib.font_manager
        from chainer_pytorch_migration import chainermn
        comm = chainermn.create_communicator()
        devices = [torch.device("cuda:" + str(comm.intra_rank))]
        chainer_devices = [chainer.get_device("@cupy:" + str(comm.intra_rank))]
        chainer_devices[0].use()
    else:
        if args.devices == '':
            devices = [torch.device("cpu")]
        else:
            devices = list(map(torch.device, args.devices.split(',')))
        print('# Devices: {}'.format(",".join(map(str, devices))))
    #devices[0].use()
    device = devices[0]

    if not args.chainermn or comm.rank == 0:
        print('# epoch: {}'.format(args.epoch))
        print('# conditional: {}'.format(args.conditional))
        print('# order_preserving: {}'.format(args.preserve_order))
        print('# steps: {}'.format(args.steps))
        print('')

    train_h5f = h5py.File(os.path.join(args.dataset, "train.h5"), 'r')
    test_h5f = h5py.File(os.path.join(args.dataset, "test.h5"), 'r')

    if not args.chainermn or comm.rank == 0:
        train = formulanet.Dataset(symbols.symbols, train_h5f)
        test = formulanet.Dataset(symbols.symbols, test_h5f)
    else:
        train, test = None, None

    if args.chainermn:
        # XXX: h5py.File cannot be distributed
        if comm.rank == 0:
            train._h5f = None
            #test._h5f = None
        train = chainermn.scatter_dataset(train, comm)
        #test = chainermn.scatter_dataset(test, comm)
        # We assume train and test are chainer.datasets.SubDataset.
        train._dataset._h5f = train_h5f
        #test._dataset._h5f = test_h5f

    train_loader = torch.utils.data.DataLoader(train, collate_fn=formulanet.convert, shuffle=True, batch_size=args.batchsize)
    test_loader = torch.utils.data.DataLoader(test, collate_fn=formulanet.convert, shuffle=False, batch_size=args.batchsize)

    model = formulanet.FormulaNet(vocab_size=len(symbols.symbols), steps=args.steps,
                                  order_preserving=args.preserve_order, conditional=args.conditional)

    if len(devices) == 1:
        model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=devices)

    # "We train our networks using RMSProp [47] with 0.001 learning rate and 1 × 10−4 weight decay.
    # We lower the learning rate by 3X after each epoch."
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=10 ** (-4))

    if args.chainermn:
        w_model = cpm.links.TorchModule(model)
        w_model.to_device(chainer_devices[0])
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
        optimizer.setup(w_model)

    def loss(y_pred, y):
        assert len(y_pred) % len(y) == 0
        return F.cross_entropy(y_pred, y.repeat(len(y_pred) // len(y)))

    trainer = ignite.engine.create_supervised_trainer(
        model, optimizer, loss, device=device,
        prepare_batch=formulanet.prepare_batch)

    def output_transform(args):
        y_pred, y = args
        return y_pred[-len(y):], y

    if not args.chainermn or comm.rank == 0:
        evaluator = ignite.engine.create_supervised_evaluator(
            model,
            metrics={
                'accuracy': ignite.metrics.Accuracy(output_transform),
                'loss': ignite.metrics.Loss(loss),
            },
            device=device,
            prepare_batch=formulanet.prepare_batch)

    models = {'main': model}
    optimizers = {'main': optimizer}
    manager = ppe.training.IgniteExtensionsManager(
        trainer, models, optimizers, args.epoch, out_dir=args.out)

    # Add a bunch of extensions
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / 3.0)
    trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, LRScheduler(scheduler))

    if not args.chainermn or comm.rank == 0:
        manager.extend(extensions.LogReport())
        manager.extend(extensions.PrintReport(
            ['epoch', 'train/loss', 'val/loss', 'val/acc']))
        manager.extend(extensions.PlotReport(
            ['train/loss', 'val/loss'], x_key='epoch', file_name='loss.png'))
        manager.extend(extensions.PlotReport(
            ['val/acc'], x_key='epoch', file_name='accuracy.png'))
        manager.extend(extensions.ProgressBar(update_interval=10))
        manager.extend(extensions.IgniteEvaluator(
            evaluator, test_loader, model, progress_bar=True))
        # manager.extend(extensions.snapshot_object(
        #     model, filename='model_epoch-{.updater.epoch}'))

    if not args.chainermn:
        writer = extensions.snapshot_writers.SimpleWriter()
        snapshot = extensions.snapshot(filename='snapshot_{.updater.iteration}', n_retains=2, writer=writer)
        manager.extend(snapshot, trigger=(100, 'iteration'))

    if args.resume:
        # Resume from a snapshot
        state = torch.load(args.snapshot)
        manager.load_state_dict(state)

    @trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
    def report_loss(engine):
        ppe.reporting.report({'train/loss': engine.state.output})
    
    trainer.run(train_loader, max_epochs=args.epoch)

    if not args.chainermn or comm.rank == 0:
        torch.save(model.state_dict(), os.path.join(args.out, 'model_final.pt'))


if __name__ == '__main__':
    main()
