import argparse
import datetime
import glob
from mdutils.mdutils import MdUtils
import os
import pandas as pd

from collections import defaultdict

def iterate_subdir(input_dir, output, hw, ts_version):
    if not os.path.isdir(input_dir):
        return None

    models = defaultdict(list)
    for subdir in sorted(os.listdir(input_dir)):
        index = subdir.rfind('_', 0, subdir.rfind('_') - 1)
        models[subdir[0:index]].append(subdir)

    mdFile = MdUtils(file_name=output, title='TorchServe Benchmark on {}'.format(hw))
    mdFile.new_header(level=1,
                      title='Date: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    mdFile.new_header(level=1, title='TorchServe Version: {}'.format(ts_version))

    for model in models.keys():
        mdFile.new_header(level=2, title=model)
        files = os.path.join(input_dir, f'{model}_*', 'ab_report.csv')
        files = glob.glob(files)
        files.sort()
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        version_col = []
        for i in range(len(df.values.tolist())):
            version_col.append(ts_version)
        df.insert(0, "version", version_col, True)

        list_of_strings = list(df.columns)

        for row in df.values.tolist():
            list_of_strings.extend(row)

        mdFile.new_table(columns=len(df.columns), rows=len(df.index)+1, text=list_of_strings, text_align='center')

    mdFile.create_md_file()

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

    parser.add_argument(
        "--hw",
        action="store",
        help="the type of host, eg. CPU, GPU, or m6i.4xlarge",
    )

    parser.add_argument(
        "--branch",
        action="store",
        help="the branch or version of TorchServe",
    )
    arguments = parser.parse_args()
    iterate_subdir(arguments.input, arguments.output, arguments.hw, arguments.branch)

if __name__ == "__main__":
    main()


