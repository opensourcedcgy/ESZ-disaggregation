# -*- coding: utf-8 -*-
"""Refrigerator classifier"""
import sys
import os
home_dir = os.path.realpath('..')
sys.path.append(home_dir)
import logging
from classifiers.classifier import Classifier
import platform


class RefrigeratorClassifier(Classifier):

    def __init__(self, name='REFRIGERATOR', model_name=None, **kws):
        self.name = name
        self.model_name = model_name
        self.model_path = '/models/{}/{}/1/'.format(self.name.lower(), model_name)
        self.prob = 0.5
        self.post_processing_steps = ['fridge_online_check']
        self.sample_period_seconds = 6
        self.seq_len = 512
        # self.post_processing_steps = []

        self.net_dict = {}
        self.target_scale = 200
        self.sample_period = '6s'
        self.num_seq_per_batch = 1000
        self.stride = 50

        self.disaggregate_power = None
        self.appliance_energy = None
        self.one_instance_per_meter = True

        # classifier specific
        self.min_on_threshold = 30
        self.max_on_threshold = 200
        self.min_on_duration = 300
        self.max_on_duration = 3600

        self.activations_min_off_duration = 600
        self.activations_on_threshold = 5
        self.activations_min_on_duration = 300
        self.activations_border = 1

        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
