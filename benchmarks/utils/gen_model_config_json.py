import argparse
import copy
import json
import os
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

MODEL_CONFIG_KEY = {
    "batch_size",
    "batch_delay",
    "url",
    "requests",
    "concurrency",
    "workers",
    "input",
    "processors"
}

def convert_yaml_to_json(yaml_file_path, output_dir):
    print("convert_yaml_to_json yaml_file_path={}, output_dir={}".format(yaml_file_path, output_dir))
    with open(yaml_file_path, 'r') as f:
        yaml_dict = yaml.safe_load(f)

        for model, config in yaml_dict.items():
            for mode, mode_config in config.items():
                model_name = mode + "_" + model
                benchmark_config = {}
                batch_size_list = None
                processors = None
                workers_list = None
                for key, value in mode_config.items():
                    if key == "batch_size":
                        batch_size_list = value
                    elif key == "processors":
                        processors = value
                    elif key == "workers":
                        workers_list = value
                    elif key in MODEL_CONFIG_KEY:
                        benchmark_config[key] = value

                batch_worker_list = []
                for batch_size in batch_size_list:
                    for workers in workers_list:
                        batch_worker_list.append({
                            "batch_size" : batch_size,
                            "workers" : workers
                        })

                benchmark_configs = []
                for batch_worker in batch_worker_list:
                    benchmark_config["batch_size"] = batch_worker["batch_size"]
                    benchmark_config["workers"] = batch_worker["workers"]
                    benchmark_configs.append(copy.deepcopy(benchmark_config))

                for bConfig in benchmark_configs:
                    for i in range(len(processors)):
                        if type(processors[i]) is str:
                            path = '{}/{}'.format(output_dir, processors[i])
                            if not os.path.isdir(path):
                                continue

                            benchmark_config_file = '{}/{}_w{}_b{}.json'\
                                .format(path, model_name, bConfig["workers"], bConfig["batch_size"])
                            with open(benchmark_config_file, "w") as outfile:
                                json.dump(bConfig, outfile, indent=4)
                        elif type(processors[i]) is dict:
                            path = '{}/gpu'.format(output_dir)
                            if not os.path.isdir(path):
                                continue

                            bConfig["gpus"] = processors[i]["gpus"]
                            benchmark_config_file = '{}/{}_w{}_b{}.json'\
                                .format(path, model_name, bConfig["workers"], bConfig["batch_size"])
                            with open(benchmark_config_file, "w") as outfile:
                                json.dump(bConfig, outfile, indent=4)
                            del bConfig["gpus"]

if __name__ == "__main__":
    main()
