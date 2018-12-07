import argparse

import tensorflow as tf


# Swap types
REGULAR = "regular"
COMPOUND = "compound"

def create_argument_parser():
    argument_parser = argparse.ArgumentParser()

    # Data arguments
    argument_parser.add_argument("training_path")
    argument_parser.add_argument("validation_path")
    argument_parser.add_argument("--max-training-features", type=int, default=None, help="Maximum number of features to train on")
    argument_parser.add_argument("--max-validation-features", type=int, default=None, help="Maximum number of features used for validation")

    # Logging arguments
    argument_parser.add_argument("--log-file", type=argparse.FileType("w", encoding="utf-8"), default=None, help="File to log training and validation progress to")
    argument_parser.add_argument("-v", "--verbose", action="store_true", help="Prints training and validation progress to the terminal")

    # Feed forward arguments
    argument_parser.add_argument("-e", "--embedding-size", type=int, default=300, help="Number of dimensions of the embedding matrix")
    argument_parser.add_argument("-b", "--batch-size", type=int, default=1024, help="Maximum batch size")
    argument_parser.add_argument("-l", "--layers", default=2 * [
        tf.layers.Dense(512, use_bias=True, activation=tf.nn.relu)
    ], help='layers in json format, e.g. [{"neurons" : 512, "activation" : "relu", "neurons" : 512, "activation" : "relu"}]')
    argument_parser.add_argument("--input-dropout", type=float, default=1, help="Dropout keep probability applied on the NN input")
    argument_parser.add_argument("--layer-dropout", type=float, default=1, help="Dropout keep probability applied after each hidden layer")
    argument_parser.add_argument("--learning-rate", type=float, default=0.01, help="(Initial) learning rate for the optimizer")

    # Oracle arguments
    argument_parser.add_argument("-u", "--unlabeled", action="store_true", help="Ignore labels")
    argument_parser.add_argument("--linkage", action="store_true", help="linkage nodes and edges")
    argument_parser.add_argument("--implicit", action="store_true", help="implicit nodes and edges")
    argument_parser.add_argument("--remote", action="store_false", help="remote edges")
    argument_parser.add_argument("--swap", choices=(REGULAR, COMPOUND), default=REGULAR, help="swap transitions")
    argument_parser.add_argument("--max-swap", type=int, default=15, help="if compound swap enabled, maximum swap size")
    argument_parser.add_argument("--node-labels", action="store_false", help="prediction of node labels, if supported by format")
    argument_parser.add_argument("--use-gold-node-labels", action="store_true", help="gold node labels when parsing")
    argument_parser.add_argument("--verify", action="store_true", help="check for oracle reproducing original passage")


    return argument_parser
