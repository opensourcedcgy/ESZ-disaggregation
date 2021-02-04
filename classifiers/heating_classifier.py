# -*- coding: utf-8 -*-
"""General heating classifier"""
import numpy as np
import pandas as pd
import sys
import os
home_dir = os.path.realpath('..')
sys.path.append(home_dir)
import time
import logging
from classifiers.classifier import Classifier
from utils.edge_detection import event_detection


class HeatingClassifier(Classifier):

    def __init__(self, name='HEATING', model_name=None, **kws):
        self.name = name
        self.model_name = model_name
        self.stride = 50
        self.num_seq_per_batch = 1000
        self.seq_len = 10
        self.sample_period = '6s'
        self.prob = 0.5 # just for the sake of consistency
        self.post_processing_steps = []

        self.net_dict = {}
        self.target_scale_dict = {}
        self.disaggregate_power = None
        self.appliance_energy = None
        self.one_instance_per_meter = True

        self.transitions = None
        self.state_threshold = 50  # for event detection
        self.buffer_size = 20

        # some classifier specific parameters
        self.min_on_threshold = 800
        self.max_on_threshold = 3500
        self.min_on_duration = 18
        self.max_on_duration = 300
        self.max_on_duration_large = 1800 # 30 minutes
        self.filter_freq = True
        self.max_usages = 3

        self.activations_min_off_duration = 60 # since we allow 3 usages within a 15 minute window, atleat 1 minutes before multiple usages
        self.activations_on_threshold = 20
        self.activations_min_on_duration = 15
        self.activations_border = 1

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)

        self.sample_period_seconds = int(self.sample_period.split('s')[0])

    def load_model(self):
        pass

    def preprocess(self, measurements, events):
        self.log.info("-----Beginning preprocessing {}-----".format(self.name))
        if events is not None and (len(events) > 0):
            # smaller heaters
            classifier_small = (events['Duration'] >= self.min_on_duration) & (
                events['Duration'] <= self.max_on_duration) & (events['Average power'] <= self.max_on_threshold) & (
                               events['Average power'] >= self.min_on_threshold)

            # large heaters
            classifier_large = (events['Duration'] >= self.min_on_duration) & \
                               (events['Duration'] <= self.max_on_duration_large) & \
                               (events['Average power'] >= self.max_on_threshold)

            classified_events = events[classifier_small | classifier_large]
            count_activations = classified_events.groupby(classified_events['T1 Time'].dt.month).count()['Average power']
            if count_activations.min() >= 1:
                self.log.info("{} was detected.".format(self.name))
                self.log.info("-----Completed preprocessing {}-----".format(self.name))
                return True

        self.log.info("{} was not detected.".format(self.name))
        self.log.info("-----Completed preprocessing {}-----".format(self.name))
        return False

    def predict(self, measurements, events, freq=6000, **kwargs):

        output_power = {}
        if (events is None):
            self.log.debug("Events not provided with the arguments.")
            self.log.debug("Starting to calculate events")
            events = event_detection(measurements)
            self.log.debug("Completed calculating events")
        else:
            self.log.debug("Events were provided with the arguments.")
            self.log.info("Starting classification of water heating events.")
        mains = measurements.copy()
        if events is not None and len(events)>0:
            self.log.debug(("Number of events detected is {}.".format(len(events))))
            classify_start_time = time.time()
            heating_events = simple_kettle_classifier(events,
                                                      min_on_threshold=self.min_on_threshold,
                                                      max_on_threshold=self.max_on_threshold,
                                                      min_on_duration=self.min_on_duration,
                                                      max_on_duration=self.max_on_duration,
                                                      filter_freq=self.filter_freq,
                                                      max_usages=self.max_usages)

            classify_end_time = time.time()
            self.log.debug("Time taken for classifying heating events {} seconds.".format(
                classify_end_time-classify_start_time))
            self.log.info("Completed classification of water heating events.")

            if len(heating_events) > 0:
                predictions = pd.Series(index=mains.index, data=np.zeros(mains.shape[0]))
                for _, event in heating_events.iterrows():
                    predictions[event['T1 Time']:event['T2 Time']] = event['Average power']
                predictions.name = self.name
                output_power[self.model_name] = predictions

        self.log.info("Updating the Resume Time {}.".format(self.name))
        return output_power


def simple_kettle_classifier(events, min_on_threshold=900, max_on_threshold=3200, min_on_duration=36,
                             max_on_duration=300, max_on_duration_large=1800, filter_freq=True, max_usages=3):
    """

    :param events: pd.DataFrame, detected events from event_detection()
    :param min_on_threshold: float
    :param max_on_threshold: float
    :param min_on_duration: float
    :param max_on_duration: float
    :param max_on_duration_large: float
    :param filter_freq: boolean, specifies whether we want to filter out occassions when the kettle events
                        happen more than max_usages in any 15 minute interval.
                        Important to avoid kettle events to get confused with tumbler or oven.
    :param max_usages: int, maximum number of usages of kettle allowed in the interval
    :return:
    """
    usage_window = 900  # seconds, interval within which the maximum usages are restricted

    events.index = np.arange(len(events))
    # calculating the average power. Actually could also be just interpolation between start and end. But this also works
    events['Average power'] = (np.abs(events['T1 Active']) + np.abs(events['T2 Active'])) / 2
    # duration of the activation
    events['Duration'] = (events['T2 Time'] - events['T1 Time']).dt.total_seconds()
    # applying simple rules for classifying events for kettle:
    classifier_small = (events['Duration'] >= min_on_duration) & \
                       (events['Duration'] <= max_on_duration) & \
                       (events['Average power'] <= max_on_threshold) & \
                       (events['Average power'] >= min_on_threshold)
    # Todo: check if space heating gets confused here
    # larger heaters:
    classifier_large = (events['Duration'] >= min_on_duration) & \
                       (events['Duration'] <= max_on_duration_large) & \
                       (events['Average power'] >= max_on_threshold)
    # Filtering out events which meet the condition
    events = events[classifier_small | classifier_large]
    # making sure index is in order
    events.index = np.arange(len(events))
    num_events = len(events)
    # filter out activations where more than max_usages happen with
    if filter_freq:
        removal_list = []  # stores the transition indices which will be removed
        idx = 0
        while idx < num_events - 1:
            current_transition_end = events.loc[idx]['T2 Time']
            next_transition_start = events.loc[idx + 1]['T1 Time']
            current = idx
            N = 1  # number of usages/activations in the current usage window
            current_removal_list = []  # stores the transition indices that need to be removed for the current window
            while ((next_transition_start - current_transition_end).total_seconds() <= usage_window) and \
                    ((next_transition_start - current_transition_end).total_seconds() >= 0) and \
                    (current + N <= num_events-1):
                current_removal_list.append(current + N - 1)
                N += 1
                if (current + N) <= num_events-1:
                    next_transition_start = events.loc[current + N]['T1 Time']
            if N > max_usages:
                # removing the last activation from this sequence,note N was increased by 1 so we do N-1
                current_removal_list.append(current + N - 1)
                removal_list += current_removal_list
            idx = current + 1
    # removing events which are in the removal_list:
        events = events[~events.index.isin(removal_list)]

    kettle_events = events[['T1 Time', 'T2 Time', 'Average power']]
    return kettle_events





