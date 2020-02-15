# -*- coding: utf-8 -*-

# to avoid "_tkinter.TclError: no display name and no $DISPLAY environment variable" error
import matplotlib as mpl

mpl.use('Agg')

import argparse
import h5py
import numpy as np
import pandas as pd
import sys
import torch
import ignite

import formulanet
import symbols

sys.setrecursionlimit(10000)


def main():
    parser = argparse.ArgumentParser(description='pytorch formulanet test')

    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device specifier.')
    parser.add_argument('--dataset', '-i', default="holstep",
                        help='HDF5 file')
    parser.add_argument('--out', '-o',
                        help='output CSV file')
    parser.add_argument('--model', '-m', default='',
                        help='Saved model file')
    parser.add_argument('--conditional', action='store_true', help='Use contional model')
    parser.add_argument('--preserve-order', action='store_true', help='Use order-preserving model')
    parser.add_argument('--steps', type=int, default="3", help='Number of update steps')

    args = parser.parse_args()

    device = torch.device(args.device)

    print('# Device: {}'.format(str(device)))
    print('# conditional: {}'.format(args.conditional))
    print('# order_preserving: {}'.format(args.preserve_order))
    print('# steps: {}'.format(args.steps))
    print('')

    test_h5f = h5py.File(args.dataset, 'r')
    test = formulanet.Dataset(symbols.symbols, test_h5f)
    test_loader = torch.utils.data.DataLoader(test, collate_fn=formulanet.convert, shuffle=False, batch_size=args.batchsize)
    print(len(test))

    model = formulanet.FormulaNet(vocab_size=len(symbols.symbols), steps=args.steps,
                                  order_preserving=args.preserve_order, conditional=args.conditional)
    model.load_state_dict(torch.load(args.model))
    model.to(device)

    def output_transform(args):
        y_pred, y = args
        return y_pred[-len(y):], y

    evaluator = ignite.engine.create_supervised_evaluator(
        model,
        metrics={
            'accuracy': ignite.metrics.Accuracy(output_transform=output_transform),
            'precision': ignite.metrics.Precision(output_transform=output_transform),
            'recall': ignite.metrics.Recall(output_transform=output_transform),
            'fbeta': ignite.metrics.Fbeta(output_transform=output_transform, beta=1.0),
        },
        device=device,
        prepare_batch=formulanet.prepare_batch)

    logits = []
    expected = []

    @evaluator.on(ignite.engine.Events.ITERATION_COMPLETED)
    def save_logits(engine):
        y_pred, y = engine.state.output
        y_pred = y_pred[-len(y):]
        logits.append(y_pred.data.cpu().numpy())
        expected.append(y.data.cpu().numpy())

    evaluator.run(test_loader)

    logits = np.concatenate(logits)
    expected = np.concatenate(expected)
    n1 = np.sum(expected)
    support = [len(test) - n1, n1]

    df = pd.DataFrame({"logits_false": logits[:, 0], "logits_true": logits[:, 1], "expected": expected})
    df.to_csv(args.out, index=False)

    print("accuracy: {}".format(evaluator.state.metrics['accuracy']))
    print("precision: {}".format(evaluator.state.metrics['precision'][1]))
    print("recall: {}".format(evaluator.state.metrics['recall'][1]))
    print("F beta score: {}".format(evaluator.state.metrics['fbeta']))
    print("support: {}".format(support))

if __name__ == '__main__':
    main()
