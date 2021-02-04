# -*- coding: utf-8 -*-
"""General utilities"""
import pandas as pd


def convert_to_datetime(unix_time):
    """
    Convert unix epoch millisecond time to pandas datetime
    :param unix_time: unix epoch millisecond
    :return:
    """
    return pd.to_datetime(unix_time, unit='ms')
