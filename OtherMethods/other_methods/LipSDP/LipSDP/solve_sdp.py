import argparse
import numpy as np
import matlab.engine
from scipy.io import savemat
import os
import pprint 

def main(args):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.join(file_dir, 'matlab_engine'))
    eng.addpath(os.path.join(file_dir, r'matlab_engine/weight_utils'))
    eng.addpath(os.path.join(file_dir, r'matlab_engine/error_messages'))
    eng.addpath(os.path.join(file_dir, r'examples/saved_weights'))

    network = {
        'alpha': matlab.double([args.alpha]),
        'beta': matlab.double([args.beta]),
        'weight_path': args.weight_path,
    }

    lip_params = {
        'formulation': args.form,
        'split': matlab.logical([args.split]),
        'parallel': matlab.logical([args.parallel]),
        'verbose': matlab.logical([args.verbose]),
        'split_size': matlab.double([args.split_size]),
        'num_neurons': matlab.double([args.num_neurons]),
        'num_workers': matlab.double([args.num_workers]),
        'num_dec_vars': matlab.double([args.num_decision_vars])
    }

    L = eng.solve_LipSDP(network, lip_params, nargout=1)

    print(f'LipSDP-{args.form.capitalize()} gives a Lipschitz constant of %.03f' % L)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--form',
        default='neuron',
        const='neuron',
        nargs='?',
        choices=('neuron', 'network', 'layer', 'network-rand', 'network-dec-vars'),
        help='LipSDP formulation to use')

    parser.add_argument('-v', '--verbose',
        action='store_true',
        help='prints CVX output from solve if supplied')

    parser.add_argument('--alpha',
        type=float,
        default=0,
        nargs=1,
        help='lower bound for slope restriction bound')

    parser.add_argument('--beta',
        type=float,
        default=1,
        nargs=1,
        help='lower bound for slope restriction bound')

    parser.add_argument('--num-neurons',
        type=int,
        default=100,
        nargs=1,
        help='number of neurons to couple for LipSDP-Network-rand formulation')

    parser.add_argument('--split',
        action='store_true',
        help='splits network into subnetworks for more efficient solving if supplied')

    parser.add_argument('--parallel',
        action='store_true',
        help='parallelizes solving for split formulations if supplied')

    parser.add_argument('--split-size',
        type=int,
        default=2,
        nargs=1,
        help='number of layers in each subnetwork for splitting formulations')

    parser.add_argument('--num-workers',
        type=int,
        default=0,
        nargs=1,
        help='number of workers for parallelization of splitting formulations')

    parser.add_argument('--num-decision-vars',
        type=int,
        default=10,
        nargs=1,
        help='specify number of decision variables to be used for LipSDP')

    parser.add_argument('--weight-path',
        type=str,
        required=True,
        nargs=1,
        help='path of weights corresponding to trained neural network model')

    args = parser.parse_args()

    if args.parallel is True and args.num_workers[0] < 1:
        raise ValueError('When you use --parallel, --num-workers must be an integer >= 1.')

    if args.split is True and args.split_size[0] < 1:
        raise ValueError('When you use --split, --split-size must be an integer >= 1.')

    main(args)
