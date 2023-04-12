import argparse
import os

from utils.report import METRICS_VALIDATED, Report
from utils.update_artifacts import (
    BENCHMARK_ARTIFACTS_PATH,
    BENCHMARK_REPORT_FILE,
    BENCHMARK_REPORT_PATH,
)


def metric_valid(key, obs_val, exp_val, threshold):
    # In case of throughput, higher is better
    # In case of memory, lower is better.
    # We ignore lower values for memory related metrices
    lower = False
    if key != "throughput":
        lower = True
    return check_if_within_threshold(exp_val, obs_val, threshold) or (
        (obs_val < exp_val and lower)
    )


def check_if_within_threshold(value1, value2, threshold):
    if float(value1) == 0.0:
        return True
    return abs((value1 - value2) / float(value1)) <= threshold


def validate_reports(artifacts_dir, report_dir):
    # Read baseline reports
    baseline_reports = {}
    for _d in sorted(os.listdir(artifacts_dir)):
        dir = os.path.join(artifacts_dir, _d)
        for subdir in sorted(os.listdir(dir)):
            csv_file = os.path.join(dir, subdir, BENCHMARK_REPORT_FILE)

            report = Report()
            report.read_csv(csv_file)
            if subdir not in baseline_reports:
                baseline_reports[subdir] = report
            else:
                baseline_reports[subdir].update(report)

    # Read generated reports
    generated_reports = {}
    for subdir in sorted(os.listdir(report_dir)):
        if os.path.isdir(os.path.join(report_dir, subdir)):
            csv_file = os.path.join(report_dir, subdir, BENCHMARK_REPORT_FILE)
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
                    f"Expected value: {baseline_reports[model].properties[key]:.2f},"
                    f"Observed value: {report.properties[key]:.2f}"
                )
                error = True
        if not error:
            print(f"Model {model} successfully validated")

    if error:
        raise Exception("Failures in benchmark validation")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-artifacts-dir",
        nargs="?",
        help="directory where benchmark artifacts have been saved",
        type=str,
        default=BENCHMARK_ARTIFACTS_PATH,
    )

    parser.add_argument(
        "--input-report-dir",
        nargs="?",
        help="directory where current benchmark report is saved",
        type=str,
        default=BENCHMARK_REPORT_PATH,
    )

    args = parser.parse_args()
    validate_reports(args.input_artifacts_dir, args.input_report_dir)


if __name__ == "__main__":
    main()
