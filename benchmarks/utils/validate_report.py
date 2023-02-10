import argparse
import os

from report import Report

BENCHMARK_REPORT_CSV = "ab_report.csv"


def validate_reports(input_dir, output):
    if not os.path.isdir(input_dir):
        print("No report found")
        return -1

    for subdir in sorted(os.listdir(input_dir)):
        if os.path.isdir(os.path.join(input_dir, subdir)):
            csv_file = os.path.join(input_dir, subdir, BENCHMARK_REPORT_CSV)
            print(f"Reading {csv_file}")
            report = Report(csv_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        action="store",
        help="the dir of a list of model benchmark result subdir ",
    )

    parser.add_argument(
        "--output",
        action="store",
        help="the file path of final report ",
    )

    arguments = parser.parse_args()
    validate_reports(arguments.input, arguments.output)


if __name__ == "__main__":
    main()
