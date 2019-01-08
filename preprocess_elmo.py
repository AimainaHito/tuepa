import os
import sys

from tuepa.util.config import get_preprocess_parser, get_oracle_parser, save_args, load_args, ARGS_FILENAME, LABELS_FILENAME, \
    DEP_FILENAME, EDGE_FILENAME, POS_FILENAME
from tuepa.util.numberer import Numberer
from tuepa.data.elmo import preprocess_dataset, specific_elmo

from elmoformanylangs import Embedder

def preprocess(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("Processing passages...", file=sys.stderr)

    elmo_embedder = Embedder(args.elmo_path, batch_size=30)

    label_numberer = Numberer()
    pos_numberer = Numberer(first_elements=["<PAD>"])
    dep_numberer = Numberer(first_elements=["<PAD>"])
    edge_numberer = Numberer(first_elements=["<PAD>"])

    print("starting processing training data..", )
    training_data = preprocess_dataset(
        args.training_path,
        args,
        label_numberer=label_numberer,
        pos_numberer=pos_numberer,
        dep_numberer=dep_numberer,
        edge_numberer=edge_numberer,
        train=True
    )
    print("finished processing training data..", )

    args.num_labels = label_numberer.max
    args.num_edges = edge_numberer.max
    args.num_pos = pos_numberer.max
    args.num_deps = dep_numberer.max

    print("...starting to write training features", )
    training_shapes = specific_elmo(training_data, elmo_embedder, args, train=True)
    print("finished writing training data..", )

    # Preprocess validation set
    print("starting to process validation data..", )
    try:
        validation_data = preprocess_dataset(
        args.validation_path,
        args,
        shapes=training_shapes,
        label_numberer=label_numberer,
        pos_numberer=pos_numberer,
        dep_numberer=dep_numberer,
        edge_numberer=edge_numberer,
        train=False
        )
        print("finished processing validation data..", )

        print("..starting to write validation data..", )
        validation_shapes = specific_elmo(validation_data, elmo_embedder, args, train=False)
        print("..finished writing validation data", )

        args.label_list = label_numberer.num2value
        args.dep_list = dep_numberer.num2value
        args.edge_list = dep_numberer.num2value
        args.pos_list = dep_numberer.num2value
        args.shapes = training_shapes

        save_args(args, args.save_dir)
        # Save arguments and dictionaries
        with open(os.path.join(args.save_dir, LABELS_FILENAME), "w", encoding="utf-8") as file:
            label_numberer.to_file(file)
        with open(os.path.join(args.save_dir, DEP_FILENAME), "w", encoding="utf-8") as file:
            dep_numberer.to_file(file)
        with open(os.path.join(args.save_dir, EDGE_FILENAME), "w", encoding="utf-8") as file:
            edge_numberer.to_file(file)
        with open(os.path.join(args.save_dir, POS_FILENAME), "w", encoding="utf-8") as file:
            pos_numberer.to_file(file)
    except:
        import IPython; IPython.embed()

from ucca import constructions

if __name__ == '__main__':
    import sys

    oracle_parser = get_oracle_parser()
    constructions.add_argument(oracle_parser)
    argument_parser = get_preprocess_parser(parents=[oracle_parser])

    args = argument_parser.parse_args(sys.argv[1:])
    preprocess(args)