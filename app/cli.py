import sys
import argparse

command = sys.argv[1]
arg = sys.argv[2:]

if command == 'train':
    parser = argparse.ArgumentParser(description="Safety prediction model.", prog='model-train')
    parser.add_argument("-d", "--data-path", help='Train directory path contains features and labels sub-directory. '
                                                  'Default: "./data"')
    parser.add_argument("-v", "--val-ratio", type=float, help="Proportion of train data used as validation. "
                                                              "Doesn't validate model if not specified.")
    parser.add_argument("-s", "--sample-size", type=int, help="Number of sample used in training the model. "
                                                              "Use all data if not specified.")
    parser.add_argument("-m", "--model-path", type=int, help='Directory to save the model. Default: "./model"')
    args = parser.parse_args(arg)
    print(args.data_path)
elif command == 'test':
    parser = argparse.ArgumentParser(description="Safety prediction model.", prog='model-predict')
    parser.add_argument("-d", "--data-path", help='Data directory path contains features sub-directory. '
                                                  'Default: "./data-test"')
    parser.add_argument("-m", "--model-path", type=int, help='Directory to load the model. Default: "./model"')
    args = parser.parse_args(arg)
    print(args.data_path)
else:
    print('Command is not supported. Available commands: train, test')


