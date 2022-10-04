import pandas_market_calendars as mcal
import torch.utils.data as data
import numpy as np
import torch
from .grouping import day_group, intra_day_list
import pandas as pd


class Schedule:
    def __init__(self, start_date, end_date, tx='America/New_York'):
        nyse = mcal.get_calendar('NYSE')
        self.schedule = nyse.schedule(start_date, end_date, tz='America/New_York')

        keys = [day.date() for day in self.schedule.index]
        self.sch_dict = dict(zip(keys, self.schedule[['market_open', 'market_close']].values.tolist()))

        self.default_open = pd.Timedelta('9:30:00')
        self.default_close = pd.Timedelta('16:00:00')
    
    def __getitem__(self, date):
        date = pd.Timestamp(date).date()
        if date in self.sch_dict:
            return self.sch_dict[date]
        else:
            day = pd.Timestamp(date, tz='America/New_York')
            return [day + self.default_open, day + self.default_close]

def filter_intraday(df, open_time, close_time):
    r'''Filter out the pre / post market data'''
    #print(f'open_time: {open_time}, close_time: {close_time}')
    #print(f't0 = {df["time"].iloc[0]}')
    df = df[(df['time'] >= open_time) & (df['time'] <= close_time)]

    return df

def filter_intradays(df_list):
    r'''Filter out the pre / post market data for each day'''
    all_dates = [df['time'].iloc[0].date() for df in df_list]
    nyse = mcal.get_calendar('NYSE')
    schedule = Schedule(all_dates[0], all_dates[-1])

    return [filter_intraday(df, *schedule[df['time'].iloc[0].date()]) for df in df_list]

class History(data.Dataset):
    def __init__(self, df):
        self.df = df

        self._init()
        self.length = len(self.df_intral_day)
    
    def _init(self):
        self.df_day = day_group(self.df)
        self.df_intral_day = intra_day_list(self.df)
        self.df_intral_day_filtered = filter_intradays(self.df_intral_day)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def get_day_idx(self, idx):
        return self.df_day.iloc[idx][['O', 'H', 'L', 'C', 'V']].values.astype(np.float64)
    
    def get_full_intraday_idx(self, idx):
        r'''Return the intraday data for the idx day
        
        pre/post-market data is included
        '''
        return self.df_intral_day[idx][['O', 'H', 'L', 'C', 'V']].values.astype(np.float64)
    
    def get_intraday_idx(self, idx):
        df = self.df_intral_day_filtered[idx]
        return df[['O', 'H', 'L', 'C', 'V']].values.astype(np.float64)
    
    def get_day_range_idx(self, idx_begin, idx_end):
        r'''Return the day data for the range [idx_begin, idx_end)

        Note: idx_end is not included, but idx_begin is included
        '''
        return self.df_day.iloc[idx_begin:idx_end][['O', 'H', 'L', 'C', 'V']].values.astype(np.float64)
    
    def get_day_past_idx(self, idx, dt):
        r'''Return the day data for the range [idx - past, idx)

        Note: idx is included
        '''
        return self.get_day_range_idx(idx - dt + 1, idx + 1)
    
    def get_day_future_idx(self, idx, dt):
        r'''Return the day data for the range [idx, idx + future)

        Note: idx is included
        '''
        return self.get_day_range_idx(idx + 1, idx + dt + 1)


class TradingHistory(History):
    def __init__(self, df, window_past=16, window_future=5):
        super().__init__(df)
        self.window_past = window_past
        self.window_future = window_future
    
    def __len__(self):
        return len(self.df_day) - self.window_past - self.window_future
    
    def __getitem__(self, idx):
        r'''Returing the data for the idx day, and the past and future data
        
        The output contains the following data:
        - OHLCV data for the idx day, time interval is 5 minutes
        - OHLCV data for the past window_past days, time interval is 1 day
        - OHLCV data for the future window_future days, time interval is 1 day
        '''

        idx = idx + self.window_past
        
        t_today = self.get_intraday_idx(idx)
        t_past = self.get_day_past_idx(idx, self.window_past)
        t_future = self.get_day_future_idx(idx, self.window_future)

        t_today = torch.from_numpy(t_today).float()
        t_past = torch.from_numpy(t_past).float()
        t_future = torch.from_numpy(t_future).float()

        return t_today, t_past, t_future