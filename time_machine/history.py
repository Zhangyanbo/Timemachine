import pandas_market_calendars as mcal
import torch.utils.data as data
import numpy as np
import torch
from .grouping import day_group, intra_day_list


def filter_intraday(df, open_time, close_time):
    r'''Filter out the pre / post market data'''
    df = df[(df['time'] >= open_time) & (df['time'] <= close_time)]

    return df

def align_dates(schedule, all_dates):
    r'''remove dates that are not in the all_dates
    '''
    all_dates = set(all_dates)
    return schedule[schedule.index.isin(all_dates)]

def filter_intradays(df_list):
    r'''Filter out the pre / post market data for each day'''
    all_dates = [df['time'].iloc[0].date() for df in df_list]
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=all_dates[0], end_date=all_dates[-1], tz='America/New_York')

    schedule = align_dates(schedule, all_dates)

    return [filter_intraday(df, open_time, close_time) for df, (open_time, close_time) in zip(df_list, schedule[['market_open', 'market_close']].values)]

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
        return self.get_day_range_idx(idx, idx + dt)


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
        
        tn = self.get_intraday_idx(idx)
        tp = self.get_day_past_idx(idx, self.window_past)
        tf = self.get_day_future_idx(idx, self.window_future)

        tn = torch.from_numpy(tn).float()
        tp = torch.from_numpy(tp).float()
        tf = torch.from_numpy(tf).float()

        return tn, tp, tf