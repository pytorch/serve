import argparse
import os

def iterate_subdir(benchmark_report_dir):
    model_benchmark_dict = {}
    if not os.path.isdir(benchmark_report_dir):
        return None

    for root, subdirs, files in os.walk(benchmark_report_dir):
        for file in files:
            if file == "ab_report.csv":


def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        action="store",
        help="the dir of a list of model benchmark subdir ",
    )

    arguments = parser.parse_args()
    iterate_subdir(arguments.input)

if __name__ == "__main__":
    main()


