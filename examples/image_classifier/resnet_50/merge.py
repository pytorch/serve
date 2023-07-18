import argparse
import os

import pandas as pd


def merge(dir):
    df_csv_append = pd.DataFrame()

    # append the CSV files
    csv_files = [os.path.join(dir, file) for file in os.listdir(dir)]

    print(csv_files)

    df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

    df.to_csv("benchmark_report.csv", index=False)

    df1 = df.pivot(index="Batch size", columns="Model Name", values="TS throughput")

    df1.to_csv("throughput.csv")

    df2 = df.pivot(index="Batch size", columns="Model Name", values="TS latency mean")

    df2.to_csv("latency_mean.csv")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir",
        action="store",
        help="Benchmark report MD",
    )
    args = parser.parse_args()

    merge(args.dir)


if __name__ == "__main__":
    main()
