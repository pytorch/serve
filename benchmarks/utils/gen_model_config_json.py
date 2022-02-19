import argparse
import copy
import json
import yaml


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        action="store",
        help="model benchmark config yaml file path",
    )
    parser.add_argument(
        "--output",
        action="store",
        help="dir for model benchmark config json file",
    )
    arguments = parser.parse_args()
    convert_yaml_to_json(arguments.input, arguments.output)

def convert_yaml_to_json(yaml_file_path, output_dir):
    with open(yaml_file_path, 'r') as f:
        yaml_dict = yaml.safe_load(f)

        for model, config in yaml_dict.items():
            if model == "instance_types":
                continue

            for mode, mode_config in config.items():
                model_name = mode + "_" + model
                benchmark_config = {}
                batch_size_list = None
                processors = None
                for key, value in mode_config.items():
                    if key == "batch_size":
                        batch_size_list = value
                    elif key == "url":
                        benchmark_config["url"] = value
                    elif key == "requests":
                        benchmark_config["requests"] = value
                    elif key == "concurrency":
                        benchmark_config["concurrency"] = value
                    elif key == "batch_delay":
                        benchmark_config["batch_delay"] = value
                    elif key == "workers":
                        benchmark_config["workers"] = value
                    elif key == "input":
                        benchmark_config["input"] = value
                    elif key == "processors":
                        processors = value

                benchmark_configs = []
                for batch_size in batch_size_list:
                    benchmark_config["batch_size"] = batch_size
                    benchmark_configs.append(copy.deepcopy(benchmark_config))

                for bConfig in benchmark_configs:
                    for i in range(len(processors)):
                        if type(processors[i]) is str:
                            benchmark_config_file = '{}/{}/{}_b{}.json'\
                                .format(output_dir, processors[i], model_name, bConfig["batch_size"])
                            with open(benchmark_config_file, "w") as outfile:
                                json.dump(bConfig, outfile, indent=4)
                        elif type(processors[i]) is dict:
                            bConfig["gpus"] = processors[i]["gpus"]
                            benchmark_config_file = '{}/gpu/{}_b{}.json'\
                                .format(output_dir, model_name, bConfig["batch_size"])
                            with open(benchmark_config_file, "w") as outfile:
                                json.dump(bConfig, outfile, indent=4)
                            del bConfig["gpus"]

if __name__ == "__main__":
    main()
