import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


class SafetyModel:

    def __init__(self, model_type: str):
        self._model_type = model_type

    def build(self, data: pd.DataFrame, label: pd.DataFrame) -> None:
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_model_type(self) -> str:
        return self._model_type

    @staticmethod
    def preprocess_label(labels: pd.DataFrame) -> pd.DataFrame:
        return labels.groupby(['bookingID']).max().reset_index().copy(deep=False)

    @staticmethod
    def _ensure_sorted(dataset: pd.DataFrame) -> pd.DataFrame:
        dataset_copy = dataset.copy(deep=False)

        dataset_copy['sequence'] = dataset_copy[
            ['bookingID', 'second']
        ].groupby('bookingID').rank(ascending=True, method='first')

        dataset_copy = dataset_copy.sort_values(by=['bookingID', 'second'])
        return dataset_copy

    @staticmethod
    def _gyro_data_enrich(dataset: pd.DataFrame) -> pd.DataFrame:
        enriched_dataset = dataset.copy(deep=False)
        enriched_dataset = SafetyModel._ensure_sorted(enriched_dataset)

        gyro_cols = ['gyro_x', 'gyro_y', 'gyro_z']

        # Find gyroscope bias / stable values
        for col in gyro_cols:
            if (col + '_stable') in enriched_dataset.columns:
                continue
            agg_stable = enriched_dataset.groupby('bookingID')[col].mean().reset_index()
            agg_stable.columns = ['bookingID', col + '_stable']
            enriched_dataset = pd.merge(enriched_dataset, agg_stable, how='left', on='bookingID', validate='m:1', copy=False)

        # Gyroscope filtered / calibrated values
        for col in gyro_cols:
            if (col + '_filtered') in enriched_dataset.columns:
                continue
            enriched_dataset[col + '_filtered'] = enriched_dataset[col] - enriched_dataset[col + '_stable']

        # Gyroscope magnitude of calibrated values
        enriched_dataset['gyro_filtered_magnitude'] = np.sqrt(enriched_dataset['gyro_x_filtered'] ** 2 + \
                                                              enriched_dataset['gyro_y_filtered'] ** 2 + \
                                                              enriched_dataset['gyro_z_filtered'] ** 2)

        # Gyroscope magnitude standard deviation
        agg_std = enriched_dataset.groupby('bookingID')['gyro_filtered_magnitude'].std().reset_index()
        agg_std.columns = ['bookingID', 'gyro_filtered_std']
        enriched_dataset = pd.merge(enriched_dataset, agg_std, how='left', on='bookingID', validate='m:1', copy=False)

        return enriched_dataset

    @staticmethod
    def _accel_data_enrich(dataset: pd.DataFrame, smoothing: int = 3) -> pd.DataFrame:
        enriched_dataset = dataset.copy(deep=False)
        enriched_dataset = SafetyModel._ensure_sorted(enriched_dataset)

        accel_cols = pd.Series(['acceleration_x', 'acceleration_y', 'acceleration_z'])

        # Rolling mean of accleration data to find gravity
        rolling_mean_data = enriched_dataset.groupby('bookingID').apply(
            lambda x: x[
                accel_cols
            ].rolling(window=smoothing, min_periods=1, center=True).mean())
        rolling_mean_data.columns = accel_cols + '_gravity'
        enriched_dataset = pd.concat([enriched_dataset, rolling_mean_data], axis=1, verify_integrity=True, copy=False)

        # Acceleration magnitude
        enriched_dataset['acceleration_magnitude'] = np.sqrt(enriched_dataset['acceleration_x'] ** 2 + \
                                                             enriched_dataset['acceleration_y'] ** 2 + \
                                                             enriched_dataset['acceleration_z'] ** 2)

        # Current acceleration vs gravity diff
        for col in accel_cols:
            enriched_dataset[col + '_gravity_diff'] = enriched_dataset[col] - enriched_dataset[col + '_gravity']
        enriched_dataset['acceleration_gravity_diff_magnitude'] = np.sqrt(
            enriched_dataset['acceleration_x_gravity_diff'] ** 2 +
            enriched_dataset['acceleration_y_gravity_diff'] ** 2 +
            enriched_dataset['acceleration_z_gravity_diff'] ** 2)

        # Acceleration magnitude standard deviation
        agg_std = enriched_dataset.groupby('bookingID')[
            'acceleration_magnitude', 'acceleration_gravity_diff_magnitude'].std().reset_index()
        agg_std.columns = ['bookingID', 'acceleration_std', 'acceleration_gravity_diff_std']
        enriched_dataset = pd.merge(enriched_dataset, agg_std, how='left', on='bookingID', validate='m:1', copy=False)

        # Phone orientation
        # enriched_dataset['orientation_theta'] = np.arctan(enriched_dataset.acceleration_x_gravity /
        #                                                   np.sqrt(enriched_dataset.acceleration_y_gravity ** 2 +
        #                                                           enriched_dataset.acceleration_z_gravity ** 2)
        #                                                   ) / np.pi * 360
        # enriched_dataset['orientation_psi'] = np.arctan(enriched_dataset.acceleration_y_gravity /
        #                                                 np.sqrt(enriched_dataset.acceleration_x_gravity ** 2 +
        #                                                         enriched_dataset.acceleration_z_gravity ** 2)
        #                                                 ) / np.pi * 360
        # enriched_dataset['orientation_phi'] = np.arctan(
        #     np.sqrt(enriched_dataset.acceleration_x_gravity ** 2 + enriched_dataset.acceleration_y_gravity ** 2) /
        #     enriched_dataset.acceleration_z_gravity) / np.pi * 360

        return enriched_dataset

    @staticmethod
    def _diff_data_enrich(dataset: pd.DataFrame) -> pd.DataFrame:
        enriched_dataset = dataset.copy(deep=False)
        enriched_dataset = SafetyModel._ensure_sorted(enriched_dataset)

        # Construct diff
        diff_data = enriched_dataset.groupby('bookingID')['second', 'Bearing', 'Speed'].diff()
        diff_data = diff_data.rename(columns=lambda x: x + '_diff')

        # Modify Bearing diff to -180 to 180
        diff_data.Bearing_diff = diff_data.Bearing_diff
        diff_data.Bearing_diff[diff_data.Bearing_diff < -180.0] += 180
        diff_data.Bearing_diff[diff_data.Bearing_diff > 180.0] -= 180

        # Difference / second (normalization)
        diff_data['Bearing_dps'] = diff_data['Bearing_diff'] / diff_data['second_diff']
        diff_data['Speed_dps'] = diff_data['Speed_diff'] / diff_data['second_diff']

        # Combine
        diff_data = diff_data.fillna(0)
        enriched_dataset = pd.concat([enriched_dataset, diff_data], axis=1, verify_integrity=True, copy=False)

        # Combine accuracy of two sequence
        acc_sum = enriched_dataset.groupby('bookingID')['Accuracy'] \
            .rolling(window=2, min_periods=1).sum().reset_index(drop=True).tolist()
        enriched_dataset['Accuracy_sum'] = acc_sum

        return enriched_dataset

    @staticmethod
    def _preprocess(dataset: pd.DataFrame) -> pd.DataFrame:
        print('Preprocess - Gyro ...')
        dataset = SafetyModel._gyro_data_enrich(dataset)
        print('Preprocess - Accelerometer ...')
        dataset = SafetyModel._accel_data_enrich(dataset, smoothing=5)
        print('Preprocess - Diff ...')
        dataset = SafetyModel._diff_data_enrich(dataset)
        return dataset


def combine_safety_pred_label(prediction_df, label_df):
    """Combine two DataFrame, each DataFrame should contains 'bookingID' column."""
    return pd.merge(prediction_df, label_df, how='left', on='bookingID', validate='1:1', copy=False)


def evaluate_safety(prediction_df, label_df):
    """Return AUC evaluation given prediction and label DataFrame. Both should have 'bookingID' column."""
    pred_label_df = combine_safety_pred_label(
        SafetyModel.preprocess_label(prediction_df),
        SafetyModel.preprocess_label(label_df))
    return roc_auc_score(pred_label_df.label, pred_label_df.prediction)


class SafetyModelBuilder:

    def __init__(self):
        self._model_dict = {
            SafetyModelByAggregation.MODEL_TYPE: SafetyModelByAggregation
        }

    def from_persistence(self, path: str) -> SafetyModel:
        obj = joblib.load(path)
        persistence_model_type = obj['model_type']
        if persistence_model_type is None:
            raise ValueError('Loaded model is not valid, does not have model_type attribute')

        if persistence_model_type not in self._model_dict.keys():
            raise NotImplementedError('Model with type {} is not registered'.format(persistence_model_type))

        safety_model_class = self._model_dict[persistence_model_type]
        safety_model = safety_model_class()
        safety_model.load(path)
        return safety_model


class SafetyModelByAggregation(SafetyModel):
    MODEL_TYPE = 'safety-aggregation_v0'

    def __init__(self):
        super(SafetyModelByAggregation, self).__init__(self.MODEL_TYPE)
        self._features = None
        self._model = None

    def build(self, data: pd.DataFrame, label: pd.DataFrame) -> None:
        print('Preprocess data ...')
        train_label = self.preprocess_label(label)
        train_dataset_prep = SafetyModel._preprocess(data)
        print('Aggregate data ...')
        train_agg_data = self._aggregate_data(train_dataset_prep)
        train_agg_data = pd.merge(train_agg_data, train_label, on='bookingID', validate='1:1', copy=False)
        print('Building RandomForestClassifier model ...')
        self._features = train_agg_data.columns[train_agg_data.columns.str.contains("max|std|ratio")]
        self._model = RandomForestClassifier(n_estimators=100, random_state=0, min_samples_leaf=75)
        self._model.fit(train_agg_data[self._features], train_agg_data.label)

    def save(self, path: str):
        obj = {
            'model_type': self._model_type,
            'features': self._features,
            'model': self._model
        }
        joblib.dump(obj, path, protocol=2)

    def load(self, path: str):
        obj = joblib.load(path)
        if obj['model_type'] != self.MODEL_TYPE:
            raise ValueError('Incompatible type to load. Expect {} but get {}'
                             .format(self.MODEL_TYPE, obj['model_type']))
        self._features = obj['features']
        self._model = obj['model']

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise AttributeError('Model is not available. Build or load the model beforehand.')
        print('Preprocess data ...')
        test_dataset_prep = self._preprocess(data)
        print('Aggregate data ...')
        test_agg_data = self._aggregate_data(test_dataset_prep)
        print('Predicting ...')
        prediction = self._model.predict_proba(test_agg_data[self._features])
        prediction = prediction[:, np.argwhere(self._model.classes_ == 1)[0][0]]
        prediction_df = pd.DataFrame(data={'bookingID': test_agg_data.bookingID, 'prediction': prediction})
        return prediction_df

    @staticmethod
    def _aggregate_data(preprocessed_dataset: pd.DataFrame) -> pd.DataFrame:
        features_max = ['gyro_filtered_magnitude',
                        'acceleration_magnitude',
                        'Speed',
                        'Bearing_dps',
                        'Speed_dps',
                        'Accuracy_sum',
                        'second',
                        'sequence',
                        'acceleration_x_gravity_diff',
                        'acceleration_y_gravity_diff',
                        'acceleration_z_gravity_diff',
                        'acceleration_gravity_diff_magnitude',
                        'gyro_x_filtered',
                        'gyro_y_filtered',
                        'gyro_z_filtered']

        agg_max = preprocessed_dataset.groupby('bookingID')[features_max].max().reset_index()
        agg_max.columns = ['bookingID'] + (pd.Series(features_max) + '_max').tolist()

        features_std = ['gyro_filtered_magnitude', 'acceleration_gravity_diff_magnitude']
        agg_std = preprocessed_dataset.groupby('bookingID')[features_std].std().reset_index()
        agg_std.columns = ['bookingID'] + (pd.Series(features_std) + '_std').tolist()

        agg_data = pd.merge(agg_max, agg_std, on='bookingID', validate='1:1', copy=False)
        agg_data['second_sequence_ratio'] = agg_data['second_max'] / agg_data['sequence_max'].astype(float)
        return agg_data
