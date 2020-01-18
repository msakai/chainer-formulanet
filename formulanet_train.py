# -*- coding: utf-8 -*-

# to avoid "_tkinter.TclError: no display name and no $DISPLAY environment variable" error
import matplotlib as mpl

mpl.use('Agg')

import argparse
import chainer
from chainer.training import extensions
import h5py
import numpy as np
import os
import sys

import torch
import torch.nn.functional as F
import ignite
import chainer_pytorch_migration as cpm
import chainer_pytorch_migration.ignite

import formulanet
import holstep
import symbols

sys.setrecursionlimit(10000)


def main():
    parser = argparse.ArgumentParser(description='chainer formulanet trainer')

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
    args.chainermn = False

    if args.chainermn:
        # matplotlib.font_manager should be imported before mpi4py.MPI
        # to avoid MPI issue with fork() system call.
        import matplotlib.font_manager
        import chainermn
        comm = chainermn.create_communicator()
        devices = [chainer.get_device("@cupy:" + str(comm.intra_rank))]
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
            test._h5f = None
        train = chainermn.scatter_dataset(train, comm)
        test = chainermn.scatter_dataset(test, comm)
        # We assume train and test are chainer.datasets.SubDataset.
        train._dataset._h5f = train_h5f
        test._dataset._h5f = test_h5f

    train_loader = torch.utils.data.DataLoader(train, collate_fn=formulanet.convert, shuffle=True, batch_size=args.batchsize)
    test_loader = torch.utils.data.DataLoader(test, collate_fn=formulanet.convert, shuffle=False, batch_size=args.batchsize)

    model = formulanet.FormulaNet(vocab_size=len(symbols.symbols), steps=args.steps,
                                  order_preserving=args.preserve_order, conditional=args.conditional)

    if len(devices) == 1:
        model.to(devices[0])

    # "We train our networks using RMSProp [47] with 0.001 learning rate and 1 × 10−4 weight decay.
    # We lower the learning rate by 3X after each epoch."
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=10 ** (-4))

    def loss(y_pred, y):
        assert len(y_pred) % len(y) == 0
        return F.cross_entropy(y_pred, y.repeat(len(y_pred) // len(y)))

    trainer = ignite.engine.create_supervised_trainer(
        model, optimizer, loss, device=device,
        prepare_batch=formulanet.prepare_batch)

    def output_transform(y_pred, y):
        return y_pred[-len(y):], y

    evaluator = ignite.engine.create_supervised_evaluator(
        model,
        metrics={
            'accuracy': ignite.metrics.Accuracy(output_transform),
            'loss': ignite.metrics.Loss(loss),
        },
        device=device,
        prepare_batch=formulanet.prepare_batch)

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def validation(engine):
        evaluator.run(val_loader)
        average_accuracy = evaluator.state.metrics['accuracy']
        average_loss = evaluator.state.metrics['loss']
        print(average_accuracy, average_loss)

    optimizer.target = model
    trainer.out = args.out

    if args.resume:
        # Resume from a snapshot
        cpm.ignite.load_chainer_snapshot(trainer, optimizer, args.resume)

    # Add a bunch of extensions
    cpm.ignite.add_trainer_extension(trainer, optimizer, extensions.ExponentialShift(
        "lr", rate=1 / 3.0), trigger=(1, 'epoch'))
    cpm.ignite.add_trainer_extension(trainer, optimizer, extensions.LogReport())
    cpm.ignite.add_trainer_extension(trainer, optimizer, extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy',
         'elapsed_time']))
    cpm.ignite.add_trainer_extension(trainer, optimizer, extensions.PlotReport(
        ['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    cpm.ignite.add_trainer_extension(trainer, optimizer, extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    snapshot = extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}')
    if args.chainermn:
        replica_sets = [[0], range(1, comm.size)]
        snapshot = chainermn.extensions.multi_node_snapshot(comm, snapshot, replica_sets)
    cpm.ignite.add_trainer_extension(trainer, optimizer, snapshot, trigger=(1, 'epoch'))
    
    trainer.run(train_loader, max_epochs=args.epoch)

    if not args.chainermn or comm.rank == 0:
        torch.save(model.state_dict(), os.path.join(args.out, 'model_final.pt'))


if __name__ == '__main__':
    main()
