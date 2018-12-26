import argparse


# Swap types
REGULAR = "regular"
COMPOUND = "compound"

def create_argument_parser():
    argument_parser = argparse.ArgumentParser()
    # Parser for general arguments
    common_parser = argparse.ArgumentParser(add_help=False)

    # Data arguments
    common_parser.add_argument("training_path")
    common_parser.add_argument("validation_path")
    common_parser.add_argument("--max-training-features", type=int, default=None, help="Maximum number of features to train on")
    common_parser.add_argument("--max-validation-features", type=int, default=None, help="Maximum number of features used for validation")
    common_parser.add_argument("--max-training-length", type=int, default=100, help="Maximum training sentences.")

    # Logging arguments
    common_parser.add_argument("--log-file", type=argparse.FileType("w", encoding="utf-8"), default=None, help="File to log training and validation progress to")
    common_parser.add_argument("-v", "--verbose", action="store_true", help="Prints training and validation progress to the terminal")

    # Oracle arguments
    common_parser.add_argument("-u", "--unlabeled", action="store_true", help="Ignore labels")
    common_parser.add_argument("--linkage", action="store_true", help="linkage nodes and edges")
    common_parser.add_argument("--implicit", action="store_true", help="implicit nodes and edges")
    common_parser.add_argument("--remote", action="store_false", help="remote edges")
    common_parser.add_argument("--swap", choices=(REGULAR, COMPOUND), default=REGULAR, help="swap transitions")
    common_parser.add_argument("--max-swap", type=int, default=15, help="if compound swap enabled, maximum swap size")
    common_parser.add_argument("--node-labels", action="store_false", help="prediction of node labels, if supported by format")
    common_parser.add_argument("--use-gold-node-labels", action="store_true", help="gold node labels when parsing")
    common_parser.add_argument("--verify", action="store_true", help="check for oracle reproducing original passage")

    # General neural network arguments
    common_parser.add_argument("-e", "--embedding-size", type=int, default=300, help="Number of dimensions of the word embedding matrix")
    common_parser.add_argument("-b", "--batch-size", type=int, default=1024, help="Maximum batch size")
    common_parser.add_argument("--learning-rate", type=float, default=0.01, help="(Initial) learning rate for the optimizer")

    # Set up subparsers for model specific parameters
    subparsers = argument_parser.add_subparsers(dest="model_type")
    subparsers.required = True

    feed_forward_parser = subparsers.add_parser("feedforward", parents=[common_parser])
    transformer_parser = subparsers.add_parser("transformer", parents=[common_parser])
    elmo_rnn_parser = subparsers.add_parser("elmo-rnn", parents=[common_parser])

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

    elmo_rnn_parser.add_argument("-elmo", "--elmo-path", required=True, help="Path to ELMo trained with ELMoForManyLangs.")
    elmo_rnn_parser.add_argument("-ff", "--ff-path", required=True, help="Path to finalfrontier embeddings.")
    elmo_rnn_parser.add_argument("--history-embedding-size", type=int, default=300, help="Size of the action history embeddings")
    elmo_rnn_parser.add_argument("--epoch_steps", type=int, default=100, help="Batches per training / evaluation epoch")

    return argument_parser
