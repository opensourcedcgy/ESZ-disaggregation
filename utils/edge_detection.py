# -*- coding: utf-8 -*-
"""
Reference find_steady_states, MyDeque and PairBuffer from : https://github.com/nilmtk/nilmtk/blob/master/nilmtk/feature_detectors/steady_states.py
 and https://github.com/nilmtk/nilmtk/blob/236b16906ec1f6e6ba973e30af11affe5f7e2c9a/nilmtk/disaggregate/hart_85.py
"""
from __future__ import print_function, division
import pandas as pd
import numpy as np
from collections import deque


class MyDeque(deque):
    def popmiddle(self, pos):
        self.rotate(-pos)
        ret = self.popleft()
        self.rotate(pos)
        return ret


class PairBuffer(object):
    """
    Attributes:
    * transitionList (list of tuples)
    * matchedPairs (dataframe containing matched pairs of transitions)
    """

    def __init__(self, buffer_size, min_tolerance, percent_tolerance,
                 large_transition, num_measurements):
        """
        Parameters
        ----------
        buffer_size: int, optional
            size of the buffer to use for finding edges
        min_tolerance: int, optional
            variance in power draw allowed for pairing a match
        percent_tolerance: float, optional
            if transition is greater than large_transition, then use percent of large_transition
        large_transition: float, optional
            power draw of a Large transition
        num_measurements: int, optional
            2 if only active power
            3 if both active and reactive power
        """
        # We use a deque here, because it allows us quick access to start and end popping
        # and additionally, we can set a maxlen which drops oldest items. This nicely
        # suits Hart's recomendation that the size should be tunable.
        self._buffer_size = buffer_size
        self._min_tol = min_tolerance
        self._percent_tol = percent_tolerance
        self._large_transition = large_transition
        self.transition_list = MyDeque([], maxlen=self._buffer_size)
        self._num_measurements = num_measurements
        if self._num_measurements == 3:
            # Both active and reactive power is available
            self.pair_columns = ['T1 Time', 'T1 Active', 'T1 Reactive',
                                 'T2 Time', 'T2 Active', 'T2 Reactive']
        elif self._num_measurements == 2:
            # Only active power is available
            self.pair_columns = ['T1 Time', 'T1 Active',
                                 'T2 Time', 'T2 Active']
        self.matched_pairs = pd.DataFrame(columns=self.pair_columns)

    def clean_buffer(self):
        # Remove any matched transactions
        for idx, entry in enumerate(self.transition_list):
            if entry[self._num_measurements]:
                self.transition_list.popmiddle(idx)
                self.clean_buffer()
                break
                # Remove oldest transaction if buffer cleaning didn't remove anything
                # if len(self.transitionList) == self._bufferSize:
                #    self.transitionList.popleft()

    def add_transition(self, transition):
        # Check transition is as expected.
        assert isinstance(transition, (tuple, list))
        # Check that we have both active and reactive powers.
        assert len(transition) == self._num_measurements
        # Convert as appropriate
        if isinstance(transition, tuple):
            mtransition = list(transition)
        # Add transition to List of transitions (set marker as unpaired)
        mtransition.append(False)
        self.transition_list.append(mtransition)
        # checking for pairs
        # self.pairTransitions()
        # self.cleanBuffer()

    def pair_transitions(self):
        """
        Hart 85, P 33.
        When searching the working buffer for pairs, the order in which
        entries are examined is very important. If an Appliance has
        on and off several times in succession, there can be many
        pairings between entries in the buffer. The algorithm must not
        allow an 0N transition to match an OFF which occurred at the end
        of a different cycle, so that only ON/OFF pairs which truly belong
        together are paired up. Otherwise the energy consumption of the
        appliance will be greatly overestimated. The most straightforward
        search procedures can make errors of this nature when faced with
        types of transition sequences.
        Hart 85, P 32.
        For the two-state load monitor, a pair is defined as two entries
        which meet the following four conditions:
        (1) They are on the same leg, or are both 240 V,
        (2) They are both unmarked,
        (3) The earlier has a positive real power component, and
        (4) When added together, they result in a vector in which the
        absolute value of the real power component is less than 35
        Watts (or 3.5% of the real power, if the transitions are
        over 1000 W) and the absolute value of the reactive power
        component is less than 35 VAR (or 3.5%).
        ... the correct way to search the buffer is to start by checking
        elements which are close together in the buffer, and gradually
        increase the distance. First, adjacent  elements are checked for
        pairs which meet all four requirements above; if any are found
        they are processed and marked. Then elements two entries apart
        are checked, then three, and so on, until the first and last
        element are checked...
        """

        tlength = len(self.transition_list)
        pairmatched = False
        if tlength < 2:
            return pairmatched

        # Can we reduce the running time of this algorithm?
        # My gut feeling is no, because we can't re-order the list...
        # I wonder if we sort but then check the time... maybe. TO DO
        # (perhaps!).

        # Start the element distance at 1, go up to current length of buffer
        for eDistance in range(1, tlength):
            idx = 0
            while idx < tlength - 1:
                # We don't want to go beyond length of array
                compindex = idx + eDistance
                if compindex < tlength:
                    val = self.transition_list[idx]
                    # val[1] is the active power and
                    # val[self._num_measurements] is match status
                    if (val[1] > 0) and (val[self._num_measurements] is False):
                        compval = self.transition_list[compindex]
                        if compval[self._num_measurements] is False:
                            # Add the two elements for comparison
                            vsum = np.add(
                                val[1:self._num_measurements],
                                compval[1:self._num_measurements])
                            # Set the allowable tolerance for reactive and
                            # active
                            matchtols = [self._min_tol, self._min_tol]
                            for ix in range(1, self._num_measurements):
                                matchtols[ix - 1] = self._min_tol if (max(np.fabs([val[ix], compval[ix]]))
                                                                      < self._large_transition) else (self._percent_tol
                                                                                                      * max(
                                            np.fabs([val[ix], compval[ix]])))
                            if self._num_measurements == 3:
                                condition = (np.fabs(vsum[0]) < matchtols[0]) and (
                                        np.fabs(vsum[1]) < matchtols[1])

                            elif self._num_measurements == 2:
                                condition = np.fabs(vsum[0]) < matchtols[0]

                            if condition:
                                # Mark the transition as complete
                                self.transition_list[idx][
                                    self._num_measurements] = True
                                self.transition_list[compindex][
                                    self._num_measurements] = True
                                pairmatched = True

                                # Append the OFF transition to the ON. Add to
                                # dataframe.
                                matchedpair = val[0:self._num_measurements] + compval[0:self._num_measurements]
                                self.matched_pairs.loc[len(self.matched_pairs)] = matchedpair

                    # Iterate Index
                    idx += 1
                else:
                    break

        return pairmatched


def find_steady_states(dataframe, state_threshold=29, noise_level=70):
    """
    Find steady states
    :param dataframe:  pandas.DataFrame, measurements
    :param state_threshold: float, the threshold to judge if the power change is valid or not for state
    :param noise_level: float, to judge if the transition is valid or noise
    :return:
            transitions:
    """

    num_measurements = len(dataframe.columns)
    estimated_steady_power = np.array([0] * num_measurements)
    last_steady_power = np.array([0] * num_measurements)
    previous_measurement = np.array([0] * num_measurements)

    df = pd.DataFrame(index=dataframe.index, data=dataframe.values, columns=["Power"])
    df['diff'] = np.fabs(df['Power'].diff())
    df.loc[df.index[0], 'diff'] = (np.fabs(np.subtract(df.loc[df.index[0], 'Power'], previous_measurement)))
    df['diff'] = df['diff']
    df['instantaneous_change'] = (df['diff'] > state_threshold).astype(bool)
    df.loc[df.index[1]:, 'ongoing_change'] = df['instantaneous_change'].shift(1)
    df.loc[df.index[0], 'ongoing_change'] = False
    df['ongoing_change'] = df['ongoing_change'].astype(bool)

    # Todo: this will have to be adopted to take into account reactive power
    df['indicator'] = np.arange(1, df.shape[0] + 1)
    df['indicator'] = df['indicator'] * df['instantaneous_change']
    df.loc[df['indicator'] == 0, 'indicator'] = np.nan
    df.loc[df.index[0], 'indicator'] = 0
    df['indicator'].fillna(method='ffill', inplace=True)
    df = df.join(df.groupby('indicator')['Power'].mean(), on='indicator', rsuffix='_r')
    df.rename(columns={'Power_r': 'estimated_steady_power'}, inplace=True)

    # calculating last_steady_power
    df.loc[(df['instantaneous_change'] & ~df['ongoing_change']), 'last_steady_power'] = \
    df['estimated_steady_power'].shift(1)[(df['instantaneous_change'] & ~df['ongoing_change'])]
    df.loc[df.index[0], 'last_steady_power'] = 0
    df['last_steady_power'].fillna(method='ffill', inplace=True)

    # calculating transitions time
    df.loc[(df['instantaneous_change'] & ~df['ongoing_change']), 'Time'] = df[
        (df['instantaneous_change'] & ~df['ongoing_change'])].index
    df['Time'] = df['Time'].shift(1)
    df.loc[df.index[0], 'Time'] = dataframe.iloc[0].name
    df['Time'].fillna(method='ffill', inplace=True)

    # calculating last_transition
    df.loc[(df['instantaneous_change'] & ~df['ongoing_change']), 'last_transition'] = \
        (df['estimated_steady_power'] - df['last_steady_power']).shift(1)

    # appending the very last transition:
    df.loc[df.index[-1], 'last_transition'] = df.loc[df.index[-1], 'estimated_steady_power'] - \
                                              df.loc[df.index[-1], 'last_steady_power']
    df = df.loc[np.fabs(df['last_transition']) > noise_level, ['Time', 'last_transition']]
    df.set_index('Time', inplace=True)
    if len(df) > 0:
        if df.index[0] == dataframe.iloc[0].name:
            df = df.iloc[1:, :]
    transitions = pd.DataFrame(df['last_transition'])
    transitions.rename(columns={"last_transition": 'active transition'}, inplace=True)
    return transitions


def event_detection(mains):
    """
    Event detection
    :param mains: pandas.DataFrame, measurements with columns ['power'] and index in pandas datetime
    :return:
            events: pandas.DataFrame, with columns ['T1 Time', 'T2 Time', 'Average power', 'Duration']
    """
    transitions = find_steady_states(
        pd.DataFrame(data=mains.values, index=mains.index.values, columns=['Power']),
        state_threshold=30, noise_level=30)
    events = None

    if len(transitions.index) != 0:
        subset = list(transitions.itertuples())
        pair_obj = PairBuffer(buffer_size=20, min_tolerance=100, percent_tolerance=0.10,
                              large_transition=1000, num_measurements=2)
        for s in subset:
            if len(pair_obj.transition_list) == 20:
                pair_obj.clean_buffer()
            pair_obj.add_transition(s)
            pair_obj.pair_transitions()
        events = pair_obj.matched_pairs.sort_values(by='T1 Time')

        if len(events) >= 1:
            events['Average power'] = (np.abs(events['T1 Active']) + np.abs(events['T2 Active'])) / 2
            try:
                events['Duration'] = (events['T2 Time'] - events['T1 Time']).dt.total_seconds()
            except AttributeError:
                events['Duration'] = (events['T2 Time'] - events['T1 Time']).total_seconds()
        else:
            events = None
    return events
