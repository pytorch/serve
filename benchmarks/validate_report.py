import argparse
import os

from utils.report import METRICS_VALIDATED, Report
from utils.update_artifacts import BENCHMARK_ARTIFACTS_PATH

BENCHMARK_REPORT_CSV = "ab_report.csv"
CWD = os.getcwd()


def metric_valid(key, obs_val, exp_val, threshold):
    lower = False
    if key != "throughput":
        lower = True
    return check_if_within_range(exp_val, obs_val, threshold) or (
        (obs_val < exp_val and lower)
    )


def check_if_within_range(value1, value2, threshold):
    if float(value1) == 0.0:
        return True
    return abs((value1 - value2) / float(value1)) <= threshold


def validate_reports(args):
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print("No report generated")
        return -1

    print(input_dir)
    # Read baseline reports
    baseline_reports = {}
    for _d in sorted(os.listdir(input_dir)):
        dir = os.path.join(input_dir, _d)
        print(dir)
        for subdir in sorted(os.listdir(dir)):
            print(os.path.join(dir, subdir))
            csv_file = os.path.join(dir, subdir, BENCHMARK_REPORT_CSV)

            report = Report()
            report.read_csv(csv_file)
            if subdir not in baseline_reports:
                baseline_reports[subdir] = report
            else:
                baseline_reports[subdir].update(report)

    # Read generated reports
    generated_reports = {}
    for subdir in sorted(os.listdir(input_dir)):
        if os.path.isdir(os.path.join(input_dir, subdir)):
            csv_file = os.path.join(input_dir, subdir, BENCHMARK_REPORT_CSV)
            report = Report()
            report.read_csv(csv_file)
            generated_reports[subdir] = report

    # Compare generated reports with baseline reports
    error = False
    for model, report in generated_reports.items():
        for key in METRICS_VALIDATED:
            if not metric_valid(
                key,
                report.properties[key],
                baseline_reports[model].properties[key],
                baseline_reports[model].deviation,
            ):
                print(
                    f"Error while validating {key} for model: {model}, "
                    f"Expected value: {baseline_reports[model].properties[key]},"
                    f"Observed value: {report.properties[key]}"
                )
                error = True
    if error:
        return -1
    print(f"Model {model} successfully validated")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        nargs="?",
        help="the dir of a list of model benchmark result subdir ",
        type=str,
        default=BENCHMARK_ARTIFACTS_PATH,
    )

    parser.add_argument(
        "--input_cfg",
        action="store",
        help="benchmark config yaml file path",
    )

    arguments = parser.parse_args()
    validate_reports(arguments)


if __name__ == "__main__":
    main()
