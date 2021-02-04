# -*- coding: utf-8 -*-
"""
Run disaggregation for all selected appliances
"""
from __future__ import print_function
import os.path
import sys

sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

select_gpu = 0
mem_limit = 2500
# tf train setup
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Create 2 virtual GPUs with 1GB memory each
    try:
        tf.config.experimental.set_visible_devices(gpus[select_gpu], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[select_gpu], True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[select_gpu],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

import pandas as pd
import importlib
import logging
import time
from utils.utils import convert_to_datetime
from config import default_models
import platform
from utils.edge_detection import event_detection


logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

_module_names = {
    'AIR_CONDITIONER': 'air_conditioner_classifier',
    'BASE_LOAD': 'base_load_classifier',
    'DISHWASHER': 'dishwasher_classifier',
    'DRYER': 'dryer_classifier',
    'ELECTRIC_VEHICLE': 'electric_vehicle_classifier',
    'HEATING': 'heating_classifier',
    'REFRIGERATOR': 'refrigerator_classifier',
    'WASHING_MACHINE': 'washing_machine_classifier',

}

_classifier_names = {
    'AIR_CONDITIONER': 'AirConditionerClassifier',
    'BASE_LOAD': 'BaseLoadClassifier',
    'DISHWASHER': 'DishwasherClassifier',
    'DRYER': 'DryerClassifier',
    'ELECTRIC_VEHICLE': 'ElectricVehicleClassifier',
    'HEATING': 'HeatingClassifier',
    'REFRIGERATOR': 'RefrigeratorClassifier',
    'WASHING_MACHINE': 'WashingMachineClassifier',
}

app_name = {
    'AIR_CONDITIONER': 'AirConditioner',
    'BASE_LOAD': 'BaseLoad',
    'DISHWASHER': 'Dishwasher',
    'DRYER': 'Dryer',
    'ELECTRIC_VEHICLE': 'ElectricVehicle',
    'HEATING': 'Heating',
    'REFRIGERATOR': 'Refrigerator',
    'WASHING_MACHINE': 'WashingMachine',
}


class Disaggregator(object):

    def __init__(self):

        self.models_dir = './models'
        self.default_appliance_model_dict = default_models
        self.appliance_classifier_dict = {}  # {Key: appliance name, Value: Classifier Object}

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        self.load_default_models()

    def load_default_models(self):
        """
        Loads the default models into the memory
        """

        self.log.info("-----Starting to load default models for all the appliances-----")
        for appliance_name, model_name in self.default_appliance_model_dict.items():
            if not self._check_if_model_already_loaded(appliance_name, model_name):
                try:
                    module = importlib.import_module(('classifiers.{}'.format(_module_names[appliance_name])))
                    class_ = getattr(module, '{}'.format(_classifier_names[appliance_name]))
                    self.appliance_classifier_dict[appliance_name] = class_(model_name=model_name)
                except Exception as e:
                    raise e

                try:
                    self.log.info("Loading model with model name {} for appliance {}......".format(
                        model_name, appliance_name))
                    self.appliance_classifier_dict[appliance_name].load_model()
                except ImportError as e:
                    raise e

        self.log.info("-----Completed loading default models-----")

    def _check_if_model_already_loaded(self, appliance_name, model_name):
        if appliance_name in self.appliance_classifier_dict:
            if self.appliance_classifier_dict[appliance_name].model_name == model_name:
                return True
            return False
        return False

    def _load_models(self, appliance_name, model_name):
        if not self._check_if_model_already_loaded(appliance_name, model_name):
            self.log.info("Model not found. Loading model {} for appliance {}.".format(model_name, appliance_name))
            module = importlib.import_module(('classifiers.{}'.format(_module_names[appliance_name])))
            class_ = getattr(module, '{}'.format(_classifier_names[appliance_name]))
            self.appliance_classifier_dict[appliance_name] = class_(model_name=model_name)
            try:
                self.appliance_classifier_dict[appliance_name].load_model()
            except ImportError as e:
                self.log.exception(e)
                raise e
        else:
            self.log.info("Model {} already loaded for appliance {}".format(model_name, appliance_name))

    def _select_measurements(self, measurements, resumeTime):
        """
        Select measurements from input by resumeTime
        :param measurements:
        :param resumeTime:
        :return:
        """
        if isinstance(resumeTime, int):
            return measurements.loc[convert_to_datetime(resumeTime):, :]
        elif isinstance(resumeTime, pd.datetime):
            return measurements.loc[resumeTime:, :]
        else:
            raise TypeError("Resume time not an integer, long or a valid pd.datetime: {}".format(resumeTime))

    def predict(self, measurements, algoDefinitions, freq, meter=None):
        """
        Predict all disaggregation results for selected appliances.
        :param measurements: pandas.DataFrame, measurements with columns ['power'] and index in pandas datetime
        :param algoDefinitions:  List<algorithmId, appliance, resumeTime>,
        :param freq: int, frequency of measurements
        :param meter: str, meter name
        :return:
                output: dict, appliance_name as key and appliance_power_dict as value
        """
        self.log.info("-----Starting event detection on {}-----".format(meter))

        if freq > 10 * 1000:
            # do not resample for these meters with larger resolution
            sample_period = '60s'
            measurements = measurements.resample(sample_period).mean()  # .resample('900s').mean()
        else:
            sample_period = '6s'
            measurements = measurements.resample(sample_period).mean()
        measurements = measurements.fillna(0)

        output = {}
        if len(measurements) > 10:
            events = event_detection(measurements)

            for algo_def in algoDefinitions:
                # Retreiving all the information
                appliance_name = algo_def['applianceName']
                algorithm_id = algo_def.get("algorithmId")
                resumeTime = algo_def.get('resumeTime')

                df = self._select_measurements(measurements, resumeTime)
                self.log.info("Loading models.........")
                self._load_models(appliance_name, algorithm_id)
                appliance_classifier = self.appliance_classifier_dict[appliance_name]

                appliance_present = appliance_classifier.preprocess(measurements=df, events=events)
                # Todo: Correct this, appliance_present= True
                self.log.info("Is Appliance {} present: {}.".format(appliance_name, appliance_present))
                if appliance_present:
                    # appliance_power = pd.DataFrame()
                    appliance_power_dict = appliance_classifier.predict(df, events, freq)
                output[appliance_name] = appliance_power_dict

        return output


def algorithm_select(appliance_list, mains):
    """
    Select appliances for prediction
    :param appliance_list: list, list of selected appliances
    :param mains:
    :return:
    """

    if isinstance(mains.index[0], int):
        start_time = convert_to_datetime(mains.index[0])
    elif isinstance(mains.index[0], pd.datetime):
        start_time = mains.index[0]
    else:
        raise TypeError("Resume time not an integer, long or a valid pd.datetime: {}".format(type(mains.index[0])))
    algo_def = [
        {"algorithmId": 'exp1',
         "resumeTime": start_time,
         "applianceName": "REFRIGERATOR"
         },
        {"algorithmId": 'exp1',
         "resumeTime": start_time,
         "applianceName": "WASHING_MACHINE"
         },
        {"algorithmId": 'exp1',
         "resumeTime": start_time,
         "applianceName": "DISHWASHER"
         },
        {"algorithmId": 'simple',
         "resumeTime": start_time,
         "applianceName": "HEATING"
         },
        {"algorithmId": 'simple',
         "resumeTime": start_time,
         "applianceName": "BASE_LOAD"
         },
        {"algorithmId": 'exp1',
         "resumeTime": start_time,
         "applianceName": "DRYER"
         },
    ]

    new_algo_def = []
    for item in algo_def:
        if item['applianceName'] in appliance_list:
            new_algo_def.append(item)
    return new_algo_def


def test(plot=True):

    mains = pd.read_csv(
        './dataset/' + 'building_1_mains.csv',
        header=None, index_col=None)
    building_1_dish_washer = pd.read_csv(
        './dataset/' + 'building_1_dish_washer.csv',
        header=None, index_col=None)
    building_1_fridge = pd.read_csv(
        './dataset/' + 'building_1_fridge.csv',
        header=None, index_col=None)
    building_1_kettle = pd.read_csv(
        './dataset/' + 'building_1_kettle.csv',
        header=None, index_col=None)
    building_1_washing_machine = pd.read_csv(
        './dataset/' + 'building_1_washing_machine.csv',
        header=None, index_col=None)

    ground_truth = {
        'DISHWASHER': building_1_dish_washer,
        'HEATING': building_1_kettle,
        'REFRIGERATOR': building_1_fridge,
        'WASHING_MACHINE': building_1_washing_machine, }

    df = mains.copy()
    df.columns = ['power']
    df.index = pd.date_range(start='2017-01-01 00:00:00', periods=len(df), freq='6s')
    print('length of df {}'.format(len(df)))
    print(df.tail())
    print("--------------------")
    print(building_1_dish_washer.head())

    algoDefinitions = algorithm_select(['BASE_LOAD', 'DISHWASHER',
                                        'HEATING', 'REFRIGERATOR', 'WASHING_MACHINE'], df)
    container = Disaggregator()
    events_start = time.time()
    print("Time taken {}".format(time.time() - events_start))
    print("-----Starting Predictions-----")
    start = time.time()
    predicted = container.predict(df, algoDefinitions, freq=1000)
    print("TF Prediction done in {} with {}".format(time.time() - start, predicted))

    if plot:
        import matplotlib.pyplot as plt
        # plot predicted
        plt.figure(figsize=(16, 8))
        legend = []
        for app, pre_dict in predicted.items():
            if app != 'BASE_LOAD':
                for model_name, pre_series in pre_dict.items():
                    plt.plot(pre_series.index, pre_series.values)
                    legend.append('Predicted: {}'.format(model_name + '_' + pre_series.name))
        plt.plot(df.index, df['power'].values)
        legend.append('Input')
        plt.xlabel('Time')
        plt.ylabel('Leistung [W]')
        plt.legend(legend, loc='upper right')
        plt.title('All_disaggregation_test', wrap=True)
        plt.savefig('All_disaggregation_test.png')

        # plot ground truth
        plt.figure(figsize=(16, 8))
        legend = []
        for app, true_y in ground_truth.items():
            plt.plot(df.index, true_y.values)
            legend.append('Ground truth: {}'.format(app))

        plt.plot(df.index, df['power'].values)
        legend.append('Input')
        plt.xlabel('Time')
        plt.ylabel('Leistung [W]')
        plt.legend(legend, loc='upper right')
        plt.title('All ground truth', wrap=True)
        plt.savefig('All_ground_truth.png')

        # plot app-wise ground truth and predicted
        for app, pre_dict in predicted.items():
            if app != 'BASE_LOAD':
                plt.figure(figsize=(16, 8))
                legend = []
                true_y = ground_truth[app]
                plt.plot(df.index, true_y.values)
                legend.append('Ground truth: {}'.format(app))
                for model_name, pre_series in pre_dict.items():
                    plt.plot(pre_series.index, pre_series.values)
                    legend.append('Predicted: {}'.format(pre_series.name))

                plt.xlabel('Time')
                plt.ylabel('Leistung [W]')
                plt.legend(legend, loc='upper right')
                plt.title('Ground_Truth vs Predicted: {}'.format(app), wrap=True)
                plt.savefig('Ground_Truth_vs_Predicted_{}.png'.format(app))
        # plt.show()

        # plot all predicted together
        plt.figure(figsize=(16, 8))
        legend = []
        for app, pre_dict in predicted.items():
            for model_name, pre_series in pre_dict.items():
                plt.plot(pre_series.index, pre_series.values)
                legend.append('Predicted: {}'.format(model_name + '_' + app))
        plt.xlabel('Time')
        plt.ylabel('Leistung [W]')
        plt.legend(legend, loc='upper right')
        plt.title('All Predicted', wrap=True)
        plt.savefig('All_Predicted.png')
        # plt.show()


if __name__ == "__main__":
    test()
