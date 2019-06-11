import sys
import argparse
import pandas as pd

from sklearn.metrics import roc_auc_score

from app.common.utils import read_csv_from_folder
from app.model.safety import SafetyModelByAggregation


def combine_safety_pred_label(prediction_df, label_df):
    """Combine two DataFrame, each DataFrame should contains 'bookingID' column."""
    return pd.merge(prediction_df, label_df, how='left', on='bookingID', validate='1:1')


def evaluate_safety(prediction_df, label_df):
    """Return AUC evaluation given prediction and label DataFrame. Both should have 'bookingID' column."""
    pred_label_df = combine_safety_pred_label(prediction_df, label_df)
    return roc_auc_score(pred_label_df.label, pred_label_df.prediction)


def train_and_validate(data_path: str, model_file_path: str, sample_size: str = None, val_ratio: float = None):
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
    model = SafetyModelByAggregation()
    print('Start to build model with {} bookings ...'.format(len(train_dataset.bookingID.unique())))
    model.build(train_dataset, train_label)
    print('Saving model to {}'.format(model_file_path))
    model.save(model_file_path)

    # Validation
    if val_ratio is not None:
        print('Validation to {} bookings'.format(len(val_dataset.bookingID.unique())))
        prediction_df = model.predict(val_dataset)
        print('Validation AUC score: {}'.format(evaluate_safety(prediction_df, val_label)))

    print('Done')


if __name__ == "__main__":
    command = sys.argv[1]
    arg = sys.argv[2:]

    if command == 'train':
        parser = argparse.ArgumentParser(description="Safety prediction model.", prog='model-train')
        parser.add_argument("-d", "--data-dir",
                            help='Train directory path contains features and labels sub-directory.',
                            default='./data')
        parser.add_argument("-v", "--val-ratio", type=float,
                            help="Proportion of train data used as validation. Doesn't validate model if not specified.")
        parser.add_argument("-s", "--sample-size", type=int,
                            help="Number of sample used in training the model. Use all data if not specified.")
        parser.add_argument("-m", "--model-file", help='Directory to save the model.',
                            default='./model/safety_model_0.mdl')
        args = parser.parse_args(arg)

        train_and_validate(args.data_dir, args.model_file, args.sample_size, args.val_ratio)

    elif command == 'test':
        parser = argparse.ArgumentParser(description="Safety prediction model.", prog='model-predict')
        parser.add_argument("-d", "--data-path", help='Data directory path contains features sub-directory. '
                                                      'Default: "./data-test"')
        parser.add_argument("-m", "--model-path", type=int, help='Directory to load the model. Default: "./model"')
        args = parser.parse_args(arg)
        print(args.data_path)
    else:
        print('Command is not supported. Available commands: train, test')


