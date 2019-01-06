import argparse
import copy
import os
import pickle

from ucca import constructions

ARGS_FILENAME = "args.pickle"

# Swap types
REGULAR = "regular"
COMPOUND = "compound"

def create_argument_parser():
    argument_parser = argparse.ArgumentParser()
    # Parser for general arguments
    common_parser = argparse.ArgumentParser(add_help=False)

    # Data arguments
    common_parser.add_argument("training_path", help="Glob to UCCA annotated training data")
    common_parser.add_argument("validation_path", help="Glob to UCCA annotated validation data")
    common_parser.add_argument("--max-training-features", type=int, default=None, help="Maximum number of features to train on")
    common_parser.add_argument("--max-validation-features", type=int, default=None, help="Maximum number of features used for validation")
    common_parser.add_argument("--max-training-length", type=int, default=100, help="Maximum training sentences.")

    # Logging arguments
    common_parser.add_argument("--log-file", type=argparse.FileType("w", encoding="utf-8"), default=None, help="File to log training and validation progress to")
    common_parser.add_argument("-v", "--verbose", action="store_true", help="Prints training and validation progress to the terminal")

    # Saving arguments
    common_parser.add_argument("--save-dir", default=None, help="directory the trained neural network model will be saved to every epoch")

    # General neural network arguments
    common_parser.add_argument("-e", "--embedding-size", type=int, default=300, help="Number of dimensions of the word embedding matrix")
    common_parser.add_argument("-b", "--batch-size", type=int, default=1024, help="Maximum batch size")
    common_parser.add_argument("--learning-rate", type=float, default=0.01, help="(Initial) learning rate for the optimizer")

    # Oracle arguments
    oracle_parser = argparse.ArgumentParser(add_help=False)

    oracle_parser.add_argument("-u", "--unlabeled", action="store_true", help="Ignore labels")
    oracle_parser.add_argument("--linkage", action="store_true", help="linkage nodes and edges")
    oracle_parser.add_argument("--implicit", action="store_true", help="implicit nodes and edges")
    oracle_parser.add_argument("--remote", action="store_false", help="remote edges")
    oracle_parser.add_argument("--swap", choices=(REGULAR, COMPOUND), default=REGULAR, help="swap transitions")
    oracle_parser.add_argument("--max-swap", type=int, default=15, help="if compound swap enabled, maximum swap size")
    oracle_parser.add_argument("--node-labels", action="store_false", help="prediction of node labels, if supported by format")
    oracle_parser.add_argument("--use-gold-node-labels", action="store_true", help="gold node labels when parsing")
    oracle_parser.add_argument("--verify", action="store_true", help="check for oracle reproducing original passage")
    oracle_parser.add_argument("--constraints", action="store_false", help="scheme-specific rules")
    oracle_parser.add_argument("--require-connected", action="store_true", help="constraint that output graph must be connected")
    oracle_parser.add_argument("--max-action-ratio", type=float, default=100, help="max action/terminal ratio")
    constructions.add_argument(oracle_parser)

    # Set up subparsers for model specific parameters
    subparsers = argument_parser.add_subparsers(dest="model_type")
    subparsers.required = True

    feed_forward_parser = subparsers.add_parser("feedforward", parents=[common_parser, oracle_parser])
    transformer_parser = subparsers.add_parser("transformer", parents=[common_parser, oracle_parser])
    elmo_rnn_parser = subparsers.add_parser("elmo-rnn", parents=[common_parser, oracle_parser])
    evaluation_parser = subparsers.add_parser("evaluate", parents=[oracle_parser])
    preprocess_parser = subparsers.add_parser("preprocess", parents=[common_parser])
    preprocess_parser.add_argument("-elmo", "--elmo-path", required=True, help="Path to ELMo trained with ELMoForManyLangs.")

    # Feed forward arguments
    feed_forward_parser.add_argument("--input-dropout", type=float, default=1, help="Dropout keep probability applied on the NN input")
    feed_forward_parser.add_argument("--layer-dropout", type=float, default=1, help="Dropout keep probability applied after each hidden layer")
    feed_forward_parser.add_argument("-l", "--layers", default='[{"neurons" : 512, "activation" : "relu", "updown" : 0}, {"neurons" : 512, "activation" : "relu", "updown" : 0}]',
        help='layers in json format, e.g. [{"neurons" : 512, "activation" : "relu", "updown" : 0}, {"neurons" : 512, "activation" : "relu", "updown" : 0}]')

    # Transformer arguments
    transformer_parser.add_argument("--self-attention-neurons", type=int, default=512, help="Number of neurons in the relu layer after self attention")
    transformer_parser.add_argument("--num-heads", type=int, default=4, help="Number of self attention heads")
    transformer_parser.add_argument("--max-positions", type=int, default=256, help="Maximum number of positions embedded by the position embeddings")
    transformer_parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")

    # Elmo/RNN arguments
    elmo_rnn_parser.add_argument("-bi-rnn", "--bi-rnn-neurons", type=int, default=512,
                                 help="Neurons in the sentence bi-rnn")
    elmo_rnn_parser.add_argument("-top-rnn", "--top-rnn-neurons", type=int, default=512,
                                 help="Neurons in the uni-directional RNN stacked on the sentence bi-rnn.")
    elmo_rnn_parser.add_argument("-hist-rnn", "--history-rnn-neurons", type=int, default=512,
                                 help="Neurons in the history rnn.")
    elmo_rnn_parser.add_argument("-l", "--layers", default='[{"neurons" : 512, "activation" : "relu", "updown" : 0}, {"neurons" : 512, "activation" : "relu", "updown" : 0}]',
        help='layers in json format, e.g. [{"neurons" : 512, "activation" : "relu", "updown" : 0}, {"neurons" : 512, "activation" : "relu", "updown" : 0}]')
    elmo_rnn_parser.add_argument("--input-dropout", type=float, default=1,
                                     help="Dropout keep probability applied on the NN input")
    elmo_rnn_parser.add_argument("--layer-dropout", type=float, default=1,
                                     help="Dropout keep probability applied after each hidden layer")
    elmo_rnn_parser.add_argument("-elmo", "--elmo-path", required=True, help="Path to ELMo trained with ELMoForManyLangs.")
    elmo_rnn_parser.add_argument("-ff", "--ff-path", required=True, help="Path to finalfrontier embeddings.")
    elmo_rnn_parser.add_argument("--history-embedding-size", type=int, default=300, help="Size of the action history embeddings")
    elmo_rnn_parser.add_argument("--epoch_steps", type=int, default=100, help="Batches per training / evaluation epoch")

    # Prediction arguments
    # evaluation_parser.add_argument("arch", choices=["feedfoward", "transformer", "elmo_rnn"], help="Type of the prediction model to be loaded")
    evaluation_parser.add_argument("model_dir", help="Directory containing a trained neural network model")
    evaluation_parser.add_argument("eval_data", help="Glob to UCCA annotated dev/test data")
    evaluation_parser.add_argument("--write-scores", action="store_true", help="Whether evaluation scores should be written to a file")
    evaluation_parser.add_argument("--timeout", type=float, default=5, help="Maximum number of seconds to wait for a single passage")
    evaluation_parser.add_argument("--check-loops", action="store_true", help="Will check repeated parser states to prevent infinite loops")
    evaluation_parser.add_argument("--action-stats", default=None, help="Output CSV filename for action statistics")
    evaluation_parser.add_argument("--normalize", action="store_true", help="Apply normalizations to UCCA output")
    evaluation_parser.add_argument("--lang", help="Target language shorthand")

    return argument_parser

def save_args(args, model_dir):
    with open(os.path.join(model_dir, ARGS_FILENAME), "wb") as file:
        # Remove log file and tensorflow layers since they can't be serialized
        args = copy.copy(args)
        if "log_file" in args:
            del args.log_file
        if "json_layers" in args:
            del args.layers

        pickle.dump(args, file)


def load_args(model_dir):
    with open(os.path.join(model_dir, ARGS_FILENAME), "rb") as file:
        return pickle.load(file)