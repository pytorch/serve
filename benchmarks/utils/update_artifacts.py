import argparse
import os
import shutil

BENCHMARK_REPORT_PATH = "/tmp/ts_benchmark"
BENCHMARK_ARTIFACTS_PATH = "/tmp/ts_artifacts"
BENCHMARK_REPORT_FILE = "ab_report.csv"
WINDOW_LEN = 8
WINDOW_START = 0

# Copy BENCHMARK_REPORT_FILE to artifacts
def copy_benchmark_reports(input, output):

    for dir in os.listdir(input):
        if os.path.isdir(os.path.join(input, dir)):
            new_dir = os.path.join(output, dir)
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy(os.path.join(input, dir, BENCHMARK_REPORT_FILE), new_dir)


# Create artifacts for a period of rolling WINDOW_LEN-1 reports
def update_artifacts(args):

    finished_copying = False
    os.makedirs(args.output, exist_ok=True)
    if not os.listdir(args.output):
        new_dir = os.path.join(args.output, str(WINDOW_START))
        print(
            f"There are no artifacts. A new package needs to be created starting at {new_dir}"
        )
        # shutil.copytree(args.input, new_dir)
        copy_benchmark_reports(args.input, new_dir)

    else:
        list_dirs = sorted(map(lambda x: int(x), os.listdir(args.output)))
        for i, dir in enumerate(list_dirs):

            if i != dir:
                new_dir = os.path.join(args.output, str(i))
                print("Creating artifacts ", new_dir)
                # shutil.copytree(args.input, new_dir)
                copy_benchmark_reports(args.input, new_dir)
                rm_dir = os.path.join(args.output, str(dir))
                shutil.rmtree(rm_dir)
                print("Removing artifacts ", rm_dir)
                finished_copying = True
                break
        i += 1
        if i < WINDOW_LEN and not finished_copying:
            new_dir = os.path.join(args.output, str(i))
            print("Creating artifacts ", new_dir)
            # shutil.copytree(args.input, new_dir)
            copy_benchmark_reports(args.input, new_dir)
            rm_dir = os.path.join(args.output, str((i + 1) % WINDOW_LEN))
            print("Removing artifacts ", rm_dir)
            shutil.rmtree(rm_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        nargs="?",
        help="the dir of a list of model benchmark result subdir ",
        const=BENCHMARK_REPORT_PATH,
        type=str,
        default=BENCHMARK_REPORT_PATH,
    )

    parser.add_argument(
        "--output",
        nargs="?",
        help="the dir of model benchmark artifacts ",
        const=BENCHMARK_ARTIFACTS_PATH,
        type=str,
        default=BENCHMARK_ARTIFACTS_PATH,
    )

    args = parser.parse_args()

    update_artifacts(args)


if __name__ == "__main__":
    main()
