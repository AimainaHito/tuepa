#!/usr/bin/env python3
from enum import IntEnum
import argparse
from time import sleep
from timeit import default_timer
import shutil
import sys

from numpy import arange


HORIZONTAL = list(map(chr, range(0x258F, 0x2588-1, -1)))
VERTICAL = list(map(chr, range(0x2581, 0x2588+1, 1)))
ASCII = list(map(str, range(10))) + ["#"]


class BlockType(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1
    ASCII = 2


def fraction_to_bar(width, fraction, blocks, fill_space=False):
    full_blocks, remainder = divmod(fraction * width, 1)
    bar = (blocks[-2] * int(full_blocks)) + blocks[round(remainder * (len(blocks) - 1)) - 1]
    if fill_space:
        return bar + (" " * (width - len(bar)))

    return bar


def make_blocks(chars):
    return chars + [""]


def get_blocks(kind):
    if kind == 'h':
        return make_blocks(HORIZONTAL)
    elif kind == 'v':
        return make_blocks(VERTICAL)
    elif kind == 'ascii':
        return make_blocks(ASCII)


def print_network_progress(
        action,
        batch_index,
        num_batches,
        losses, cummulative_losses,
        accuracy, cummulative_accuracy,
        width=None
    ):

    if width is None:
        width = shutil.get_terminal_size()[0]

    data_tail = "| {}/{} | loss: {:.1f} ({:.1f}) accuracy: {:.1%} ({:.1%}) |".format(
        batch_index, num_batches, losses, cummulative_losses, accuracy, cummulative_accuracy
    )
    data_front = "| {} |".format(action)
    data_filler = " " * (width - len(data_front) - len(data_tail))
    data = data_front + data_filler + data_tail
    print(
        data
        + "\n"
        + fraction_to_bar(width, batch_index / num_batches, HORIZONTAL, fill_space=True),
        end="\033[1A\033[{}D\033[K".format(width),
        file=sys.stderr
    )


def clear_progress():
    print("\033[K\033[1B\033[K\033[2A", end="", flush=True, file=sys.stderr)


def print_iteration_info(iteration_index, train_losses, train_accuracy, validation_losses, validation_accuracy, start_time, file=None):
    print_function = print if file is None else lambda x: file.write(x + "\n")

    print_function(
        "| Epoch {} | Training loss: {:.2f}, accuracy: {:.2%} | Validation loss: {:.2f}, accuracy: {:.2%} | Δt: {:.1f}s".format(
            iteration_index,
            train_losses,
            train_accuracy,
            validation_losses,
            validation_accuracy,
            default_timer() - start_time
        )
    )


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("space", nargs="?", type=int, default=shutil.get_terminal_size()[0])
    parser.add_argument("--start", "-s", type=float, default=0)
    parser.add_argument("--end", "-e", type=float, default=1)
    parser.add_argument("--duration", "-d", type=float, default=.1)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--block-type", "-b", choices=["h", "v", "ascii"], default='h')
    args = parser.parse_args()
    args.block_type = get_blocks(args.block_type)

    print("\033[?25l", end="")
    try:
        for step in arange(args.start, args.end, args.step):
            print(
                fraction_to_bar(
                    args.space,
                    (step - args.start) / (args.end - args.start),
                    args.block_type
                ),
                end="\r"
            )
            sleep(args.duration)

        print(fraction_to_bar(args.space, args.end, args.block_type))
        sleep(args.duration)

    finally:
        print("\033[?25h", end="")


if __name__ == '__main__':
    import sys


    main(sys.argv[1:])
