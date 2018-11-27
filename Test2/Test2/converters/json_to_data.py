from datetime import datetime
import numpy as np

class JSONToData():
  def convertJson(self, json):
    time_series_key_prefix = 'Time Series'
    metadata_key = 'Meta Data'
    symbol_key = '2. Symbol'

    symbol = json[metadata_key][symbol_key]
    time_series_data = []
    for key in json:
      if time_series_key_prefix in key:
        time_series_key = key
        break

    for date, datum in json[time_series_key].items():
      time_series_datum = [date]

      # 1. open 2. high 3. low 4. close 5. volume
      for key, value in datum.items():
        time_series_datum.append(value)

      time_series_data.insert(0, time_series_datum)

    return {symbol: time_series_data}