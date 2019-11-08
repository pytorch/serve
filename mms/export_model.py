

"""
This command line interface is no longer used. Please refer to model-archiver tool for the new CLI for exporting models.
"""


def main():
    print('\033[93m'  # Red Color start
          + "mxnet-model-export is no longer supported.\n"
            "Please use model-archiver to create 1.0 model archive.\n"
            "For more detail, see: https://pypi.org/project/model-archiver"
          + '\033[0m')  # Red Color end


if __name__ == '__main__':
    main()
