import argparse
import os

from utils.report import (
    ACCEPTABLE_METRIC_DEVIATION,
    METRICS_VALIDATED,
    Report,
    metric_valid,
)
from utils.update_artifacts import (
    BENCHMARK_ARTIFACTS_PATH,
    BENCHMARK_REPORT_FILE,
    BENCHMARK_REPORT_PATH,
)


def validate_reports(artifacts_dir, report_dir, deviation):
    # Read baseline reports
    baseline_reports = {}
    num_reports = len(os.listdir(artifacts_dir))
    for _d in sorted(os.listdir(artifacts_dir)):
        dir = os.path.join(artifacts_dir, _d)
        for subdir in sorted(os.listdir(dir)):
            csv_file = os.path.join(dir, subdir, BENCHMARK_REPORT_FILE)

            report = Report(deviation, num_reports)
            report.read_csv(csv_file)
            if subdir not in baseline_reports:
                baseline_reports[subdir] = report
            else:
                baseline_reports[subdir].update(report)

    # Get the mean value each of the properties for every report
    for model, report in baseline_reports.items():
        report.mean()
        baseline_reports[model] = report

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
                    f"Expected value: {baseline_reports[model].properties[key]:.2f}, "
                    f"Observed value: {report.properties[key]:.2f}"
                )
                error = True
        if not error:
            print(f"Model {model} successfully validated")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-artifacts-dir",
        help="directory where benchmark artifacts have been saved",
        type=str,
        default=BENCHMARK_ARTIFACTS_PATH,
    )

    parser.add_argument(
        "--input-report-dir",
        help="directory where current benchmark report is saved",
        type=str,
        default=BENCHMARK_REPORT_PATH,
    )

    parser.add_argument(
        "--deviation",
        help="acceptable variation in metrics values ",
        type=float,
        default=ACCEPTABLE_METRIC_DEVIATION,
    )
    args = parser.parse_args()
    validate_reports(args.input_artifacts_dir, args.input_report_dir, args.deviation)


if __name__ == "__main__":
    main()
