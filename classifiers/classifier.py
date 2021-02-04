# -*- coding: utf-8 -*-
""" Basis for classifier """
import pandas as pd
import numpy as np
from utils.disaggregate import disaggregate, ensemble_rectangles
from tensorflow.keras.models import load_model
import logging


class Classifier(object):
    """
    Base class for implementing device/appliance classifiers
    """

    def __init__(self, name=None, model_name=None):
        """
        Initialisation
        :param name: str, name of the appliance (type), such as 'BASE_LOAD'
        :param model_name: str, the model name of loaded model of the appliance,
        like 'exp1' for the final model version 1.
        """
        self.name = name
        self.model_name = model_name
        self.model_path = None

        self.net_dict = {}

        self.min_on_duration = 0
        self.max_on_threshold = 0
        self.min_on_threshold = 0

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)

    def load_model(self):
        """
        Load tf model for selected appliance and model version.
        """
        self.log.info("Beginning to load neural network models {}.".format(self.name))
        model = load_model(self.model_path)
        self.net_dict[self.model_name] = model
        self.log.info("Loading {} completed. Details of model loaded {}".format(self.name, self.net_dict))

    def preprocess(self, measurements, events):
        """
        Pre-process to check if the appliance is presented or not.
        :param measurements: pandas.DataFrame, measurements with columns ['power'] and index in pandas datetime
        :param events: pandas.DataFrame, detected events from event_detection()
        :return: Boolean, True for presented and False for not presented
        """
        self.log.info("-----Beginning preprocessing {}-----".format(self.name))
        if events is not None and (len(events) > 0):
            classifier = (events['Duration'] >= self.min_on_duration) & \
                         (events['Average power'] <= self.max_on_threshold) & \
                         (events['Average power'] >= self.min_on_threshold)
            classified_events = events[classifier]
            count_activations = classified_events.groupby(classified_events['T1 Time'].dt.month).count()[
                'Average power']
            if count_activations.min() >= 1:
                self.log.info("{} was detected.".format(self.name))
                self.log.info("-----Completed preprocessing {}-----".format(self.name))
                return True

        self.log.info("{} was not detected.".format(self.name))
        self.log.info("-----Completed preprocessing {}-----".format(self.name))
        return False

    def predict(self, measurements, events, freq=6000, **kwargs):
        """
        Predict function for general appliance case.
        :param measurements: pandas.DataFrame, measurements with columns ['power'] and index in pandas datetime
        :param events: pandas.DataFrame, detected events from event_detection()
        :param freq: int, frequency of measurements in ms. Default, 6000ms as data from UK_DALE
        :param kwargs:
        :return:
                output_power: dict, with model_name as key and pd.Series with predicted time series as value
        """

        output_power = {}
        mains = measurements.copy()
        rectangles_dict = {}
        self.log.info("Beginning {} predictions.".format(self.name))
        for model_name, model in self.net_dict.items():
            self.log.info(
                "Predicting for model {} with freq {} target scale of {} W.".format(model, freq, self.target_scale))
            # Getting the rectangles prediction
            rectangles = disaggregate(model, mains.values, model,
                                         num_seq_per_batch=self.num_seq_per_batch,
                                         seq_len=self.seq_len, appliance=self.name,
                                         target_scale=self.target_scale, stride=self.stride)
            rectangles_dict[model] = rectangles
            self.log.info("Prediction {} for model {} completed".format(self.name, model))
            if rectangles_dict:
                self.log.info("Ensembling the results from all the models: {}.".format(self.name))
                prediction = ensemble_rectangles(rectangles_dict, self.target_scale, self.seq_len,
                                                 self.stride, self.prob, self.sample_period_seconds, mains)
                self.log.info(
                    "-----Completed ensembling. Completed Predictions for {} with Net power sum: {}-----".format(
                        self.name, np.sum(prediction)))

                if np.any(prediction != 0):
                    net_power = pd.Series(data=prediction, index=mains.index)
                    net_power.index.name = "sampled_time"
                    net_power.name = self.name
                    output_power[self.model_name] = net_power

        return output_power



