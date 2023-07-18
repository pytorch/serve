import argparse

import pandas as pd


def convert_csv(input, model):
    rows = []
    with open(input) as f:
        lines = f.readlines()[10:]
        for row in lines:
            # Get rid of leading and trailing '|'
            tmp = row[1:-2]
            # Split line and ignore column whitespace
            clean_line = [col.strip() for col in tmp.split("|")]
            # Append clean row data to rows variable
            rows.append(clean_line)
        # Get rid of syntactical sugar to indicate header (2nd row)
        rows = rows[:1] + rows[2:]
        print(rows)
    df = pd.DataFrame(rows)
    # set column names equal to values in row index position 0
    df.columns = df.iloc[0]

    # remove first row from DataFrame
    df = df[1:]

    # reset index values
    df.reset_index(drop=True, inplace=True)

    df["Batch size"] = pd.to_numeric(df["Batch size"])

    df = df.sort_values(by=["Batch size"])
    df["Model Name"] = model

    print(df.head())
    df.to_csv("./csv_files/" + model + ".csv", index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        action="store",
        help="Benchmark report MD",
    )

    parser.add_argument(
        "--model",
        action="store",
        help="model name",
    )
    args = parser.parse_args()

    convert_csv(args.input, args.model)


if __name__ == "__main__":
    main()
