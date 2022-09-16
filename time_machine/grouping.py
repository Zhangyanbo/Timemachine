import pandas as pd


def read_csv(path):
    raw = pd.read_csv(path, header=None,
                      names=['time', 'O', 'H', 'L', 'C', 'V'],
                      parse_dates=['time'])
    # set timezone in NYC
    raw['time'] = raw['time'].dt.tz_localize('America/New_York')
    return raw

def day_group(df):
    day_df = df.groupby(pd.Grouper(key='time', freq='1D')).agg(O=('O', 'first'),
                                                               H=('H', 'max'),
                                                               L=('L', 'min'),
                                                               C=('C', 'last'),
                                                               V=('V', 'sum'))
    # remove H, M, S for time
    day_df = day_df.reset_index()
    day_df = day_df.dropna()
    day_df['time'] = day_df['time'].dt.normalize()
    return day_df

def fillna(df):
    df['O'] = df['O'].fillna(method='ffill').fillna(method='bfill')
    df['H'] = df['H'].fillna(method='ffill').fillna(method='bfill')
    df['L'] = df['L'].fillna(method='ffill').fillna(method='bfill')
    df['C'] = df['C'].fillna(method='ffill').fillna(method='bfill')
    df['V'] = df['V'].fillna(0)
    return df

def min_group5(df):
    r'''Group by 5 minutes
    
    Intervals with no trades are filled with the last trade.
    '''
    min_df = df.groupby(pd.Grouper(key='time', freq='5Min')).agg(O=('O', 'first'),
                                                                 H=('H', 'max'),
                                                                 L=('L', 'min'),
                                                                 C=('C', 'last'),
                                                                 V=('V', 'sum'))
    min_df = fillna(min_df)
    min_df = min_df.reset_index()
    return min_df

def day_padding(df, begin='04:00:00', end='20:00:00'):
    r'''Add padding to the beginning and end of the day, with interval of 5 minutes.'''
    
    begin = pd.Timestamp(df['time'].iloc[0].date()) + pd.Timedelta(begin)
    end = pd.Timestamp(df['time'].iloc[-1].date()) + pd.Timedelta(end)
    # set timezone to NYC
    begin = begin.tz_localize('America/New_York')
    end = end.tz_localize('America/New_York')
    
    df = df.set_index('time')
    df = df.reindex(pd.date_range(begin, end, freq='5Min'))
    df = df.reset_index()
    # rename index to time
    df = df.rename(columns={'index': 'time'})
    df = fillna(df)
    return df

def intra_day_list(df):
    r'''Return the list of intra-day dataframes
    
    For the pre / post market, the data is padded with the last (or nearest) trade.
    '''
    days = df.groupby(pd.Grouper(key='time', freq='1D'))
    days = filter(lambda x: len(x[1]) > 0, days)
    return [day_padding(min_group5(day)) for date, day in days]