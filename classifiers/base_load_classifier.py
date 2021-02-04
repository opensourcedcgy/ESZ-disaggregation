# -*- coding: utf-8 -*-
"""Base load classifier"""
from __future__ import print_function
import sys
import os
import pandas as pd
import logging
import time
from pytz import timezone
home_dir = os.path.realpath('..')
sys.path.append(home_dir)
from classifiers.classifier import Classifier


tz = timezone('Europe/Berlin')


class BaseLoadClassifier(Classifier):

    def __init__(self, name='BASE_LOAD', model_name=None, **kws):
        self.name = name
        self.model_name = model_name
        self.baseload_probability = 0.1  # probability of the power being higher than the baseload
        self.subsample_period = '900s'
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        self.sample_period_seconds = 6

    def load_model(self):
        """
        Base_LOAD has no model to load for gpu computation
        :return:
        """
        pass

    def preprocess(self, measurements, events):
        """
        Always return true for base load.
        :param measurements:
        :param events:
        :return:
        """
        return True

    def predict(self, measurements, events, freq=6000, **kwargs):
        """
        Predict baseload base on 10 percentile of the measurements in every 15 mins, i.e. 900s.
        :param measurements: pandas.DataFrame, measurements with columns ['power'] and index in pandas datetime
        :param events: pandas.DataFrame, detected events from event_detection()
        :param freq: int, frequency of measurements in ms.
        :param kwargs:
        :return:
                output_power: dict, with model_name as key and pd.Series with predicted time series as value
        """
        # Todo: What happens if there are not enough readings in a day.
        #  So for example, if there are only 2 measurements for the last day,
        #  you cannot assign that value as the baseline for the day.

        output_power = {}
        mains = measurements.copy()

        # convert utc time to local timezone:
        if mains.index.tzinfo is None:
            mains.index = mains.index.tz_localize('UTC').tz_convert(tz)

        phases_provided = ['power']

        df = pd.DataFrame(index=mains.index)
        df['dates'] = df.index.date
        # check if the mains get interval less than 900s, here 1000s, do resample
        if freq/1000 <= 900:
            mains_sampled = mains[phases_provided].resample(self.subsample_period).mean().fillna(0)
        else:
            mains_sampled = mains[phases_provided]
        start_time = time.time()
        daily_baseload = mains_sampled.groupby(mains_sampled.index.date).quantile(0.1)
        groupby_time = time.time()
        self.log.info("Time to calculate the quantile after groupby is {} seconds".format(groupby_time-start_time))

        daily_baseload.index.name = "dates"
        prediction = df.join(daily_baseload, how='left', on='dates')
        prediction = prediction.loc[:, phases_provided]

        # since the baseload activity should finish. For no appliance should the incomplete activity be provided
        prediction.iloc[-1, :] = 0

        # converting predictions back to UTC time
        if prediction.index.tzinfo is not None:
            prediction.index = prediction.index.tz_convert('UTC').tz_localize(None)
        join_time = time.time()
        self.log.info("Time take to perform the join {} seconds for {} number of mains".format(
            join_time-groupby_time, len(prediction)))
        prediction = prediction.sum(axis=1)
        prediction.name = self.name
        output_power[self.model_name] = prediction
        return output_power





