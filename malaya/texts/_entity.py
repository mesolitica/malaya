from ._text_functions import multireplace
from ._tatabahasa import (
    date_replace,
    _money,
    _temperature,
    _distance,
    _volume,
    _duration,
    _date,
    _past_date_string,
    _now_date_string,
    _future_date_string,
    _tomorrow_date_string,
    _yesterday_date_string,
    _depan_date_string,
    _expressions,
)
from ._normalization import money, _normalize_money
from ..cluster import cluster_entities
import dateparser
import re


class _Entity_regex:
    def __init__(self, model = None):
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
            result = cluster_entities(result_model)
        else:
            result = {}
        money_ = re.findall(_money, string)
        money_ = [(s, money(s)) for s in money_]
        dates_ = re.findall(_date, string)
        past_date_string_ = re.findall(_past_date_string, string)
        now_date_string_ = re.findall(_now_date_string, string)
        future_date_string_ = re.findall(_future_date_string, string)
        yesterday_date_string_ = re.findall(_yesterday_date_string, string)
        depan_date_string_ = re.findall(_depan_date_string, string)
        tomorrow_date_string_ = re.findall(_tomorrow_date_string, string)
        dates_ = (
            dates_
            + past_date_string_
            + now_date_string_
            + future_date_string_
            + yesterday_date_string_
            + depan_date_string_
            + tomorrow_date_string_
        )
        dates_ = [multireplace(s, date_replace) for s in dates_]
        dates_ = [re.sub(r'[ ]+', ' ', s).strip() for s in dates_]
        dates_ = {s: dateparser.parse(s) for s in dates_}
        money_ = {s[0]: _normalize_money(s[1]) for s in money_}
        temperature_ = re.findall(_temperature, string)
        temperature_ = [re.sub(r'[ ]+', ' ', s).strip() for s in temperature_]
        distance_ = re.findall(_distance, string)
        distance_ = [re.sub(r'[ ]+', ' ', s).strip() for s in distance_]
        volume_ = re.findall(_volume, string)
        volume_ = [re.sub(r'[ ]+', ' ', s).strip() for s in volume_]
        duration_ = re.findall(_duration, string)
        duration_ = [re.sub(r'[ ]+', ' ', s).strip() for s in duration_]
        phone_ = re.findall(_expressions['phone'], string)
        phone_ = [re.sub(r'[ ]+', ' ', s).strip() for s in phone_]
        email_ = re.findall(_expressions['email'], string)
        email_ = [re.sub(r'[ ]+', ' ', s).strip() for s in email_]
        url_ = re.findall(_expressions['url'], string)
        url_ = [re.sub(r'[ ]+', ' ', s).strip() for s in url_]
        time_ = re.findall(_expressions['time'], string)
        time_ = [re.sub(r'[ ]+', ' ', s).strip() for s in time_]

        result['date'] = dates_
        result['money'] = money_
        result['temperature'] = temperature_
        result['distance'] = distance_
        result['volume'] = volume_
        result['duration'] = duration_
        result['phone'] = phone_
        result['email'] = email_
        result['url'] = url_
        if 'time' in result:
            result['time'].extend(time_)
        else:
            result['time'] = time_
        return result
