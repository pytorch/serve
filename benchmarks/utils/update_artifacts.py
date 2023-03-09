import argparse
import os
import shutil

BENCHMARK_REPORT_PATH = "/tmp/ts_benchmark"
BENCHMARK_ARTIFACTS_PATH = "/tmp/ts_artifacts"
BENCHMARK_REPORT_FILE = "ab_report.csv"
WINDOW_LEN = 8
WINDOW_START = 0

################################################################
# This is an example directory structure for the artifacts.
# Here, report_id 1 is missing, new report would be added under 1
# and we would remove report_id 2.
# .
# └── tmp/
#    └── ts_artifacts/
#        ├── 0/
#        │   ├── eager_mode_mnist_w4_b1/
#        │   │   └── ab_report.csv
#        │   ├── eager_mode_mnist_w4_b2/
#        │   │   └── ab_report.csv
#        │   └── ...
#        ├── 2/
#        │   ├── eager_mode_mnist_w4_b1/
#        │   │   └── ab_report.csv
#        │   ├── eager_mode_mnist_w4_b2/
#        │   │   └── ab_report.csv
#        │   └── ...
#        ├── 3/
#        │   ├── eager_mode_mnist_w4_b1/
#        │   │   └── ab_report.csv
#        │   ├── eager_mode_mnist_w4_b2/
#        │   │   └── ab_report.csv
#        │   └── ...
#        ├── ...
#        └── 6/
#            ├── eager_mode_mnist_w4_b1/
#            │   └── ab_report.csv
#            ├── eager_mode_mnist_w4_b2/
#            │   └── ab_report.csv
#            └── ...
################################################################


# Copy BENCHMARK_REPORT_FILE to artifacts
def copy_benchmark_reports(input, output):

    for dir in os.listdir(input):
        if os.path.isdir(os.path.join(input, dir)):
            new_dir = os.path.join(output, dir)
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy(os.path.join(input, dir, BENCHMARK_REPORT_FILE), new_dir)


# Save new report and delete the oldest report
def update_new_report(input_dir, output_dir, add_report_id, del_report_id):

    # Add new report
    new_dir = os.path.join(output_dir, str(add_report_id))
    print("Creating artifacts ", new_dir)
    copy_benchmark_reports(input_dir, new_dir)

    # Remove old report
    if isinstance(del_report_id, int):
        rm_dir = os.path.join(output_dir, str(del_report_id % WINDOW_LEN))
        print("Removing artifacts ", rm_dir)
        shutil.rmtree(rm_dir, ignore_errors=True)


# Create artifacts for a period of rolling WINDOW_LEN-1 reports
def update_artifacts(input_dir, output_dir):

    # Create a drectory where artifacts will be stored
    os.makedirs(output_dir, exist_ok=True)

    # Get the sorted list of existing report_ids
    list_dirs = sorted(map(lambda x: int(x), os.listdir(output_dir)))
    num_reports = len(list_dirs)

    # Initial case: When they are less than WINDOW_LEN-1 reports
    if num_reports < WINDOW_LEN - 1:
        add_report_id, del_report_id = num_reports, None
        update_new_report(input_dir, output_dir, add_report_id, del_report_id)
        return

    # When there are WINDOW_LEN - 1 reports and we want to add the new report
    # and remove the oldest report
    for i, report_id in enumerate(list_dirs):

        if i != report_id or (i + 1 == WINDOW_LEN - 1):
            if i != report_id:
                # When  report_id has a missing element in sequence
                add_report_id, del_report_id = i, report_id
            else:
                # When report_id WINDOW_LEN-1 is missing
                add_report_id, del_report_id = i + 1, (i + 2) % WINDOW_LEN
            update_new_report(input_dir, output_dir, add_report_id, del_report_id)
            break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        nargs="?",
        help="the dir of a list of model benchmark result subdir ",
        const=BENCHMARK_REPORT_PATH,
        type=str,
        default=BENCHMARK_REPORT_PATH,
    )

    parser.add_argument(
        "--output_dir",
        nargs="?",
        help="the dir of model benchmark artifacts ",
        const=BENCHMARK_ARTIFACTS_PATH,
        type=str,
        default=BENCHMARK_ARTIFACTS_PATH,
    )

    args = parser.parse_args()

    update_artifacts(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
