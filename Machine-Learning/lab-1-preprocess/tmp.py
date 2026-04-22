import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Dataset Preprocessing")
    parser.add_argument(
        "-r",
        "--data-root",
        type=str,
        default=".",
        help="The root directory of input dataset",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for loading dataset",
    )
    parser.add_argument(
        "-t", "--target-size", type=int, default=64, help="Unified image size"
    )

    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent


if __name__ == "__main__":
    main()
