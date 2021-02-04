# -*- coding: utf-8 -*-
"""Dishwasher classifier"""
from __future__ import print_function
import sys
import os
home_dir = os.path.realpath('..')
sys.path.append(home_dir)
import logging
from classifiers.classifier import Classifier
import platform

class DishwasherClassifier(Classifier):

    def __init__(self, name='DISHWASHER', model_name=None, **kws):
        self.name = name
        self.model_name = model_name
        self.model_path = '/models/{}/{}/1/'.format(self.name.lower(), model_name)
        self.prob = 0.5
        self.post_processing_steps = ['post_process_dishwasher']
        self.sample_period_seconds = 6
        self.seq_len = 1792

        self.net_dict = {}
        self.target_scale = 500
        self.sample_period = '6s'
        self.num_seq_per_batch = 1000
        self.stride = 50

        self.disaggregate_power = None
        self.appliance_energy = None
        self.one_instance_per_meter = True

        self.transitions = None
        self.state_threshold = 50  # for event detection
        self.buffer_size = 20

        # some classifier specific parameters
        self.min_on_duration = 420  # 7 minutes
        self.max_on_threshold = 4000  # watts
        self.min_on_threshold = 800  # watts
        self.min_monthly_consumption = 1

        self.activations_min_off_duration = 1800
        self.activations_on_threshold = 10
        self.activations_min_on_duration = 1800
        self.activations_border = 1

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)





