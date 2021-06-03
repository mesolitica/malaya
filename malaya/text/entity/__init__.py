# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE
import dateparser
import re
from malaya.text.function import multireplace
from malaya.text.tatabahasa import date_replace
from malaya.text.regex import (
    _money,
    _temperature,
    _distance,
    _volume,
    _duration,
    _weight,
    _date,
    _past_date_string,
    _now_date_string,
    _future_date_string,
    _yesterday_tomorrow_date_string,
    _depan_date_string,
    _expressions,
    _left_datetime,
    _right_datetime,
    _today_time,
    _left_datetodaytime,
    _right_datetodaytime,
    _left_yesterdaydatetime,
    _right_yesterdaydatetime,
    _left_yesterdaydatetodaytime,
    _right_yesterdaydatetodaytime,
)
from malaya.text.entity.food import (
    hot_ice_beverage_regex,
    fruit_juice_regex,
    unique_beverage_regex,
    total_foods_regex,
)
from malaya.text.normalization import money, _normalize_money
from malaya.cluster import cluster_tagging


class EntityRegex:
    def __init__(self, model=None):
        self._model = model

    def predict(self, string):
        """
        classify a string.

        Parameters
        ----------
        string : str
        """

        if not isinstance(string, str):
            raise ValueError('input must be a string')

        if self._model:
            result_model = self._model.predict(string)
            result = cluster_tagging(result_model)
        else:
            result = {}
        money_ = re.findall(_money, string)
        money_ = [(s, money(s)[1]) for s in money_]
        dates_ = re.findall(_date, string)
        past_date_string_ = re.findall(_past_date_string, string)
        now_date_string_ = re.findall(_now_date_string, string)
        future_date_string_ = re.findall(_future_date_string, string)
        yesterday_date_string_ = re.findall(
            _yesterday_tomorrow_date_string, string
        )
        depan_date_string_ = re.findall(_depan_date_string, string)
        dates_ = (
            dates_
            + past_date_string_
            + now_date_string_
            + future_date_string_
            + yesterday_date_string_
            + depan_date_string_
        )
        dates_ = [multireplace(s, date_replace) for s in dates_]
        dates_ = [re.sub(r'[ ]+', ' ', s).strip() for s in dates_]
        dates_ = {s: dateparser.parse(s) for s in dates_}
        money_ = {s[0]: s[1] for s in money_}
        temperature_ = re.findall(_temperature, string)
        temperature_ = [re.sub(r'[ ]+', ' ', s).strip() for s in temperature_]
        distance_ = re.findall(_distance, string)
        distance_ = [re.sub(r'[ ]+', ' ', s).strip() for s in distance_]
        volume_ = re.findall(_volume, string)
        volume_ = [re.sub(r'[ ]+', ' ', s).strip() for s in volume_]
        duration_ = re.findall(_duration, string)
        duration_ = [re.sub(r'[ ]+', ' ', s).strip() for s in duration_]
        weight_ = re.findall(_weight, string)
        weight_ = [re.sub(r'[ ]+', ' ', s).strip() for s in weight_]
        phone_ = re.findall(_expressions['phone'], string)
        phone_ = [re.sub(r'[ ]+', ' ', s).strip() for s in phone_]
        email_ = re.findall(_expressions['email'], string)
        email_ = [re.sub(r'[ ]+', ' ', s).strip() for s in email_]
        url_ = re.findall(_expressions['url'], string)
        url_ = [re.sub(r'[ ]+', ' ', s).strip() for s in url_]
        time_ = re.findall(_expressions['time'], string)
        time_ = [re.sub(r'[ ]+', ' ', s).strip() for s in time_]
        today_time_ = re.findall(_today_time, string)
        today_time_ = [re.sub(r'[ ]+', ' ', s).strip() for s in today_time_]
        time_ = time_ + today_time_

        if 'time' in result:
            time_ = list(set(time_ + result['time']))

        time_ = [multireplace(s, date_replace) for s in time_]
        time_ = [re.sub(r'[ ]+', ' ', s).strip() for s in time_]
        time_ = {s: dateparser.parse(s) for s in time_}

        left_datetime_ = [
            '%s %s' % (i[0], i[1]) for i in re.findall(_left_datetime, string)
        ]
        right_datetime_ = [
            '%s %s' % (i[0], i[1]) for i in re.findall(_right_datetime, string)
        ]
        today_left_datetime_ = [
            '%s %s' % (i[0], i[1])
            for i in re.findall(_left_datetodaytime, string)
        ]
        today_right_datetime_ = [
            '%s %s' % (i[0], i[1])
            for i in re.findall(_right_datetodaytime, string)
        ]
        left_yesterdaydatetime_ = [
            '%s %s' % (i[0], i[1])
            for i in re.findall(_left_yesterdaydatetime, string)
        ]
        right_yesterdaydatetime_ = [
            '%s %s' % (i[0], i[1])
            for i in re.findall(_right_yesterdaydatetime, string)
        ]
        left_yesterdaydatetodaytime_ = [
            '%s %s' % (i[0], i[1])
            for i in re.findall(_left_yesterdaydatetodaytime, string)
        ]
        right_yesterdaydatetodaytime_ = [
            '%s %s' % (i[0], i[1])
            for i in re.findall(_right_yesterdaydatetodaytime, string)
        ]

        datetime_ = (
            left_datetime_
            + right_datetime_
            + today_left_datetime_
            + today_right_datetime_
            + left_yesterdaydatetime_
            + right_yesterdaydatetime_
            + left_yesterdaydatetodaytime_
            + right_yesterdaydatetodaytime_
        )
        datetime_ = [re.sub(r'[ ]+', ' ', s).strip() for s in datetime_]
        datetime_ = {s: dateparser.parse(s) for s in datetime_}

        foods_ = re.findall(total_foods_regex, string)
        foods_ = [re.sub(r'[ ]+', ' ', s).strip() for s in foods_]

        drinks_ = re.findall(unique_beverage_regex, string)
        drinks_.extend(re.findall(hot_ice_beverage_regex, string))
        drinks_.extend(re.findall(fruit_juice_regex, string))
        drinks_ = [re.sub(r'[ ]+', ' ', s).strip() for s in drinks_]

        result['date'] = dates_
        result['money'] = money_
        result['temperature'] = temperature_
        result['distance'] = distance_
        result['volume'] = volume_
        result['duration'] = duration_
        result['phone'] = phone_
        result['email'] = email_
        result['url'] = url_
        result['time'] = time_
        result['datetime'] = datetime_
        result['food'] = foods_
        result['drink'] = drinks_
        result['weight'] = weight_
        return result
