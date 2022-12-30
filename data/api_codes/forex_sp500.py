# https://fred.stlouisfed.org/tags/series?t=&et=&ptic=819202&ob=pv&od=&tg=&tt=

from fredapi import Fred
import data.config as config

fred = Fred(api_key=config.fred_api_key)
personal_income_series = fred.search_by_release(175, limit=175, order_by='popularity', sort_order='desc')
personal_income_series['title']

cpi = fred.get_series("T10Y2Y", observation_start="2010-01-01", observation_end="2022-12-31")

fund_rate = fred.get_series("DFF", observation_start="2010-01-01", observation_end="2022-12-31")