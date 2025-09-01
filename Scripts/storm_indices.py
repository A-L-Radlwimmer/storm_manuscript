import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t



def compute_climatologies(input_df, t_column_name, p_column_name):
    '''
    Compute climatologies to be used in compute_si.
    Input:   input_df         ... dataframe with daily datetime index and columns containing temperature and mean sea level pressure.
             t_column_name    ... name of column containing temperatures in °C.
             p_column_name    ... name of column containing mean sea level pressures in hPa.
    Output:  T_clim ... pandas series with month-day-hour multi-index, containing temperature climatology.
             P_clim ... pandas series with month-day-hour multi-index, containing mean sea level pressure climatology.
    '''
    
    df = input_df.copy()
    
    df['T_smoothed'] = df[t_column_name].rolling('3D', center=True).mean()
    df['P_smoothed'] = df[p_column_name].rolling('3D', center=True).mean()
    df['month'] = df.index.month
    df['day'] = df.index.day
    T_clim = df.groupby(['month', 'day'])['T_smoothed'].mean()
    P_clim = df.groupby(['month', 'day'])['P_smoothed'].mean()
    
    return T_clim, P_clim


def compute_si(input_df, t_column_name, p_column_name, si_column_name):
    '''
    Compute the Storm Index.
    Input:   input_df         ... dataframe with daily datetime index and columns containing temperature and mean sea level pressure.
             t_column_name    ... name of column containing temperatures in °C.
             p_column_name    ... name of column containing mean sea level pressures in hPa.
             si_column_name   ... name of column SI will be written to.
    Output:  output_df ... input_df with additional column named 'si_column_name', containing the SI for each timestep.
    '''
    
    df = input_df.copy()
    
    # compute climatologies
    t_clim, p_clim = compute_climatologies(df, t_column_name, p_column_name)
    
    # add climatology to df
    df_index = df.index
    df['month'] = df.index.month
    df['day'] = df.index.day
    t_clim_df = t_clim.reset_index(name='T_clim')
    p_clim_df = p_clim.reset_index(name='P_clim')
    df = pd.merge(df, t_clim_df, on=['month', 'day'], how='left')
    df = pd.merge(df, p_clim_df, on=['month', 'day'], how='left')
    df.index = df_index
    
    # positive winter temperature anomalies or negative summer temperature anomalies
    df['winter'] = np.where(df['month'].isin([4, 5, 6, 7, 8, 9]), True, False)
    condition = (((df[t_column_name]-df['T_clim'] > 0) & (df['winter']==True)) | ((df[t_column_name]-df['T_clim'] < 0) & (df['winter']==False)))
    #condition = (((df[t_column_name]-df['T_clim'] > 0) & (df['winter']==True)) | ((df[t_column_name]-df['T_clim'] > 0) & (df['winter']==False)))   # only positive temperature anomalies
    df['T_anom'] = np.where(df[t_column_name].isna(), np.nan, np.where(condition, np.abs(df[t_column_name]-df['T_clim']), 0))
    # negative pressure anomalies
    df['P_anom'] = np.where(df[p_column_name].isna(), np.nan, np.where((df[p_column_name]-df['P_clim']) < 0, np.abs(df[p_column_name]-df['P_clim']), 0))
    
    # normalize anomalies by standard devition
    df['T_anom'] = df['T_anom']/df['T_anom'].std()
    df['P_anom'] = df['P_anom']/df['P_anom'].std()
    
    # compute and smooth storm index
    df[si_column_name] = df['T_anom'] * df['P_anom']
    df[si_column_name] = df[si_column_name].rolling('3D', center=True).mean()
    
    # normalize by mean storminess of storms
    mean_storminess_of_storms = df[si_column_name][df[si_column_name] > 0].mean()
    df[si_column_name] = df[si_column_name]/mean_storminess_of_storms
    
    # drop columns
    df.drop(columns=['month', 'day', 'winter', 'T_clim', 'P_clim', 'T_anom', 'P_anom'], inplace=True)
    
    output_df = df
    
    return output_df


def compute_anomalies(input_df, t_column_name, p_column_name):
    '''
    Compute only simultaneous anomalies. Write them into their own dataframe.
    Input:   input_df         ... dataframe with daily datetime index and columns containing temperature and mean sea level pressure.
             t_column_name    ... name of column containing temperatures in °C.
             p_column_name    ... name of column containing mean sea level pressures in hPa.
    Output:  output_df ... dataframe containing the anomaly time series in daily resolution.
    '''
    
    df = input_df.copy()
    
    # compute climatologies
    t_clim, p_clim = compute_climatologies(df, t_column_name, p_column_name)
    
    # add climatology to df
    df_index = df.index
    df['month'] = df.index.month
    df['day'] = df.index.day
    t_clim_df = t_clim.reset_index(name='T_clim')
    p_clim_df = p_clim.reset_index(name='P_clim')
    df = pd.merge(df, t_clim_df, on=['month', 'day'], how='left')
    df = pd.merge(df, p_clim_df, on=['month', 'day'], how='left')
    df.index = df_index
    
    # positive winter temperature anomalies or negative summer temperature anomalies
    df['winter'] = np.where(df['month'].isin([4, 5, 6, 7, 8, 9]), True, False)
    condition = (((df[t_column_name]-df['T_clim'] > 0) & (df['winter']==True)) | ((df[t_column_name]-df['T_clim'] < 0) & (df['winter']==False)))
    #condition = (((df[t_column_name]-df['T_clim'] > 0) & (df['winter']==True)) | ((df[t_column_name]-df['T_clim'] > 0) & (df['winter']==False)))   # only positive temperature anomalies
    df['T_anom'] = np.where(df[t_column_name].isna(), np.nan, np.where(condition, np.abs(df[t_column_name]-df['T_clim']), 0))
    # negative pressure anomalies
    df['P_anom'] = np.where(df[p_column_name].isna(), np.nan, np.where((df[p_column_name]-df['P_clim']) < 0, np.abs(df[p_column_name]-df['P_clim']), 0))
    
    # normalize anomalies by standard devition
    df['T_anom'] = df['T_anom']/df['T_anom'].std()
    df['P_anom'] = df['P_anom']/df['P_anom'].std()
    
    # limit to simultaneously occurring anomalies
    mask = (df['T_anom'] > 0) & (df['P_anom'] > 0)
    df['T_anom'] = np.where(mask, df['T_anom'], np.nan)
    df['P_anom'] = np.where(mask, df['P_anom'], np.nan)
    
    output_df = df[['T_anom', 'P_anom']]
    
    return output_df


def linear_fit(df, column):
    '''
    Make a linear fit on data contained in a column of a dataframe (with a datetime or numeric index).
    Input:   df         ... dataframe (with datetime index and) columns containing data I wish to fit.
             column     ... name of column containing data to be fitted.
    Output:  m             ... slope in per day unit
             b             ... offset
             p_m, p_b      ... p-values
    '''
    x = df.index
    if isinstance(x, pd.DatetimeIndex):
        x = (x - x[0]) / pd.Timedelta("1D")  # Convert to numeric (days since start)
    else:
        x = np.asarray(x)

    y = df[column].values
    
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    popt, pcov = curve_fit(lambda x, m, b: m * x + b, x, y)
    stderr = np.sqrt(np.diag(pcov))

    dof = max(0, len(x) - len(popt))
    t_vals = popt / stderr
    p_vals = [2 * (1 - t.cdf(abs(tval), dof)) for tval in t_vals]

    return popt[0], popt[1], p_vals[0], p_vals[1]
