import sys
import argparse
import pandas as pd

from app.common.utils import read_csv_from_folder
from app.model.safety import SafetyModelByAggregation, SafetyModelBuilder, evaluate_safety, SafetyModelByCnn


def train_and_validate(model_type: str, data_path: str, model_file_path: str, sample_size: str = None,
                       val_ratio: float = None):
    # Read raw data from directory
    features_raw = read_csv_from_folder('{}/features/*.csv'.format(data_path))
    labels = read_csv_from_folder('{}/labels/*.csv'.format(data_path))

    # Sampling data
    all_booking_ids = pd.Series(features_raw.bookingID.unique())
    all_booking_ids_count = all_booking_ids.shape[0]

    booking_count = all_booking_ids_count
    if sample_size is not None:
        booking_count = min(sample_size, all_booking_ids_count)
    sample_booking_ids = pd.Series(all_booking_ids.sample(booking_count, random_state=0))

    features_raw = features_raw.loc[features_raw.bookingID.isin(sample_booking_ids), :].copy(deep=False)
    labels = labels.loc[labels.bookingID.isin(sample_booking_ids), :].copy(deep=False)

    # Split to validation if specified
    train_dataset = features_raw
    train_label = labels
    val_dataset = None
    val_label = None
    if val_ratio is not None:
        validation_booking_ids = sample_booking_ids.sample(frac=val_ratio, random_state=0)
        train_dataset = features_raw.loc[~features_raw.bookingID.isin(validation_booking_ids), :].copy(deep=False)
        train_label = labels.loc[~labels.bookingID.isin(validation_booking_ids), :].copy(deep=False)
        val_dataset = features_raw.loc[features_raw.bookingID.isin(validation_booking_ids), :].copy(deep=False)
        val_label = labels.loc[labels.bookingID.isin(validation_booking_ids), :].copy(deep=False)

    # Train and save model
    if model_type == 'rf':
        clf_model = SafetyModelByAggregation()
    elif model_type == 'cnn':
        clf_model = SafetyModelByCnn()
    else:
        raise AttributeError('Invalid model type {}'.format(model_type))

    print('Start to build model with {} bookings ...'.format(len(train_dataset.bookingID.unique())))
    clf_model.build(train_dataset, train_label)
    print('Saving model to {}'.format(model_file_path))
    clf_model.save(model_file_path)

    # Validation
    if val_ratio is not None:
        print('Validation to {} bookings'.format(len(val_dataset.bookingID.unique())))
        clf_prediction_df = clf_model.predict(val_dataset)
        print('Validation AUC score: {}'.format(evaluate_safety(clf_prediction_df, val_label)))

    print('Done')


if __name__ == "__main__":
    command = sys.argv[1]
    arg = sys.argv[2:]

    if command == 'train':
        parser = argparse.ArgumentParser(description="Safety prediction model.", prog='model-train')
        parser.add_argument("model_type", help='Model type, either "rf", "cnn", or "cnn-rf-stack"',
                            choices=["rf", "cnn", "cnn-rf-stack"])
        parser.add_argument("-d", "--data-dir",
                            help='Train directory path contains features and labels sub-directory. Default: "./data"',
                            default='./data')
        parser.add_argument("-v", "--val-ratio", type=float,
                            help="Proportion of train data used as validation. Doesn't validate model if not specified.")
        parser.add_argument("-s", "--sample-size", type=int,
                            help="Number of sample used in training the model. Use all data if not specified.")
        parser.add_argument("-m", "--model-file",
                            help='Directory to save the model. Default: "./model/safety_model_0.mdl"',
                            default='./model/safety_model_0.mdl')
        args = parser.parse_args(arg)

        train_and_validate(args.model_type, args.data_dir, args.model_file, args.sample_size, args.val_ratio)

    elif command == 'test':
        parser = argparse.ArgumentParser(description="Safety prediction model.", prog='model-predict')
        parser.add_argument("-d", "--data-dir",
                            help='Data directory path contains features sub-directory. Default: "./data-test"',
                            default='./data-test')
        parser.add_argument("-m", "--model-file",
                            help='Directory to load the model. Default: "./model/safety_model_0.mdl"',
                            default='./model/safety_model_0.mdl')
        parser.add_argument("-o", "--output-file",
                            help='Prediction output file name. Default: "./output/test_prediction.csv"',
                            default='./output/test_prediction.csv')
        args = parser.parse_args(arg)

        print('Loading model from {} ...'.format(args.model_file))
        model = SafetyModelBuilder().from_persistence(args.model_file)
        print('Got model with type: {}'.format(model.get_model_type()))
        print('Loading test data from {} ...'.format('{}/features/*.csv'.format(args.data_dir)))
        test_features = read_csv_from_folder('{}/features/*.csv'.format(args.data_dir))

        print('Predicting test data with {} rows ...'.format(test_features.shape[0]))
        prediction_df = model.predict(test_features)
        prediction_df.to_csv(args.output_file, index=False)
        print('Done')

    elif command == 'evaluate':
        parser = argparse.ArgumentParser(description="Safety prediction model.", prog='model-evaluate')
        parser.add_argument("prediction_file_path", help='Path to prediction csv-file')
        parser.add_argument("test_label_file_path", help='Path to test label csv-file')
        args = parser.parse_args(arg)

        print('Load prediction from {}'.format(args.prediction_file_path))
        prediction_df = pd.read_csv(args.prediction_file_path)
        print('Prediction have {} rows.'.format(prediction_df.shape[0]))

        print('Load test label from {}'.format(args.test_label_file_path))
        test_label = pd.read_csv(args.test_label_file_path)
        print('Test label have {} rows.'.format(test_label.shape[0]))

        print('Test AUC score: {}'.format(evaluate_safety(prediction_df, test_label)))
    else:
        print('Command is not supported. Available commands: train, test')


