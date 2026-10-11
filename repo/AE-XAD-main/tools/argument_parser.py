from argparse import ArgumentParser

class DefaultConfig(object):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('-e', type=int, help='Number of epochs', default=500)
        parser.add_argument('-s', type=int, help='Seed to use')
        parser.add_argument('-net', type=str,
                            choices=['shallow', 'deep', 'conv', 'conv_deep', 'conv_deep_v2', 'conv_f2'],
                            help='Network to use')
        parser.add_argument('-l', type=str, choices=['aexad', 'mse', 'aexad_norm'], default='aexad',
                            help='Loss to use. Available options are aexad and mse')
        parser.add_argument('--no-cuda', dest='cuda', action='store_false')
        parser.add_argument('--hist', dest='si', action='store_true',
                            help='If specified, saves the weights of the model every 100 epochs')
        return parser
class RealDSParser(DefaultConfig):
    def __call__(self, parser: ArgumentParser) -> ArgumentParser:
        parser = super().__call__(parser)
        parser.add_argument('-ds', type=str, help='Dataset to use')
        parser.add_argument('-na', type=int, default=1, help='Number of anomalies for anomaly class')
        parser.add_argument('-dp', type=str, help='Dataset path', default=None)
        return parser

