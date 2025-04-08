from __future__ import print_function
from datetime import datetime
from datetime import timedelta
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import numpy as np
import re
import logging
from math import trunc
from scipy.interpolate import UnivariateSpline
import json
import sys

SECRET_PATH = 'credentials.json'

def main(input_sheet_id, input_sheet_range, holidays_sheet_id='1DDMBg8-q5SuPpeL1ZqC8vs7D9U-_b_F0dIroFBsNvEk', holidays_sheet_range='holidays!A1:E', stddev_threshold=1.8, outlier_rolling_window=13, monthly_seasonality_window = 14, wrap_monthly_seasonality=True, weekly_seasonality_window = 16, enable_verbose_logging=False):
    """
    Imports raw volume data from a Gsheet, performs cleaning, outlier detection, calculates seasonality & returns dataframes containing the output

    Args:
        input_sheet_id (str): ID of Google Sheet to import data from.
        input_sheet_range (str): Range in the GSheet which contains the data (must be supplied in A1 notation e.g. 'raw'!A1:C500).
        mau_sheet_id (str): ID
    
    Returns:
        dict: Dictionary containing 3 Pandas DataFrames: df_raw_vol for cleaned raw data, df_intra_month_seasonality & df_intra_week_seasonality
    """
    if enable_verbose_logging==True:
        FORMAT = '%(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format=FORMAT) 
        logging.getLogger('googleapicliet.discovery_cache').setLevel(logging.ERROR) # Surpress info messages from Google oauth
        logging.info('Verbose logging enabled (level=logging.INFO)')
        logging.info('Script start time: ' + datetime.now().strftime('%d/%m/%Y %H:%M:%S'))
    
    df_raw_vol = import_gsheet_to_df(input_sheet_id, input_sheet_range)
    df_raw_vol['value'] = pd.to_numeric(df_raw_vol['value'], downcast="float")

    #holidays data
    df_holidays = import_gsheet_to_df(holidays_sheet_id, holidays_sheet_range)
    
    #flag outliers & holidays
    df_raw_vol = flag_outliers(df_raw_vol, stddev_threshold, outlier_rolling_window)
    df_raw_vol = flag_holidays(df_raw_vol, df_holidays)

    # add column to flag outliers & holidays to be excluded from seasonality calculations
    df_raw_vol['exclude_from_seasonality_calcs_flg'] = False
    df_raw_vol.loc[df_raw_vol['is_stdv_stats_outlier'] == True, 'exclude_from_seasonality_calcs_flg'] = True
    df_raw_vol.loc[df_raw_vol['is_holiday'] == True, 'exclude_from_seasonality_calcs_flg'] = True

    df_intra_week_seasonality = get_intra_week_distro(df_raw_vol, weekly_seasonality_window)
    df_raw_vol = adjust_weekday_seasonality(df_raw_vol)
    df_intra_month_seasonality = get_intra_month_seasonality(df_raw_vol, value_col='seasonally_adj_temp_value', window_size=monthly_seasonality_window, wrap_spline=wrap_monthly_seasonality)


    # timestamp datatype is not JSON serialisable so need to turn date column into a string, which will then be interpreted as a date in google sheets
    #df_raw_vol['date'] = df_raw_vol['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return df_raw_vol, df_intra_month_seasonality, df_intra_week_seasonality
    #export_df_to_google_sheet(df_raw_vol, input_sheet_id, gsheet_tab_name='raw_processed', include_df_headers=True)
    #export_df_to_google_sheet(df_intra_month_seasonality, input_sheet_id, gsheet_tab_name='intra_month_seasonality', include_df_headers=True)
    #export_df_to_google_sheet(df_intra_week_seasonality, input_sheet_id, gsheet_tab_name='intra-week-seasonality', include_df_headers=True)


def adjust_weekday_seasonality(df):
    """
    Takes a df with daily volumes and adds a column with rolling inra-week seasonality & seasonally adjusted values

    Args:
        df (pandas.DataFrame): Dataframe containing daily volume data. Must contain 'date', 'business_line_alias', 'rolling_mean' & 'value' columns.
    """
    df['iso_calendar_week'] = df['date'].dt.strftime('%G-w%V')
    #create df with weekly summed mean
    weekly_sums = df.groupby(['business_line_alias', 'iso_calendar_week'])['rolling_mean'].sum()
    weekly_sums = weekly_sums.reset_index()
    weekly_sums.rename(columns={'rolling_mean':'rolling_mean_wkly_sum'}, inplace=True) 
    
    df = df.merge(weekly_sums, left_on=['business_line_alias', 'iso_calendar_week'], right_on=['business_line_alias', 'iso_calendar_week'], how='left')

    df['rolling_intra_week_distro'] = df['rolling_mean'] / df['rolling_mean_wkly_sum'] 
    df['rolling_intra_week_distro'] = df['rolling_intra_week_distro'].clip(-1, 1) # clamp min & max values to -1 or 1 otherwise we might end up with weird stuff
    df['seasonally_adj_value'] = (df['value'] * (1 / df['rolling_intra_week_distro'])) / 7
    df['seasonally_adj_temp_value'] = (df['temp_value'] * (1 / df['rolling_intra_week_distro'])) / 7

    return df


def check_orphan_groups(df_left, df_right, grouping_col, behaviour='WARN'):
    """
    INDEV: Compare two dataframes to make sure that a given grouping column (e.g. business_line_alias) from the first df has corresponding values in the second df.

    Args:
        df_left (pandas.DataFrame): Dataframe with grouping column to be compared against.
        df_right (pandas.DataFrame): Dataframe to check for missing values in the grouping column.
        grouping_col (str): Name of the grouping column (Must be the same between df_left & df_right).
        behaviour (str): Behaviour if missing values are detected (default='WARN'). Other behaviours not yet implemented TODO  

    Returns:
        (pandas.Series): A series containing all the values from the grouping column in df_left which are missing in df_right.
    """
    df_left_groups = df_left[grouping_col].unique()
    df_right_groups = df_right[grouping_col].unique()

    missing_groups = pd.Series(np.setdiff1d(df_left_groups, df_right_groups))

    if len(missing_groups) == 0:
        return

    if behaviour == 'WARN':
        missing_groups_concat = '[' + missing_groups.str.cat(sep='\', \'') + ']'
        warn_str_plural = ''
        if missing_groups.size > 1:
            warn_str_plural = 's'
        warning_string = str(missing_groups.size) + ' missing group' + warn_str_plural + ' from ' + grouping_col + ' col: ' + missing_groups_concat
        logging.warning(warning_string)
    
    return missing_groups

def export_df_to_google_sheet(df, gsheet_id, gsheet_tab_name, include_df_headers=False, tab_colour=(0.0, 0.0, 0.0), import_to_range='A1'):
    """
    Exports dataframe to google sheets with options to specify the range to import to or include column headers.

    Args:
        df (pandas.DataFrame): Dataframe to be exported to google sheets.
        gsheet_id (str): ID of the google sheet the df is to be exported to.
        gsheet_tab_name (str): Name of the tab in the gsheet to put the data (it will be created if it doesn't exist already).
        include_df_headers (bool): Optional. If true then include the df column headers in the export (False by default).
        gsheet_range (str): Optional. Specify where to place the df data in the GSheet with a cell reference in A1 notation ('A1' by default).
    """    
    if include_df_headers == True:
        df = concat_col_headers_to_df(df)
    
    create_new_sheet(gsheet_id, gsheet_tab_name, tab_colour) #does nothing if a sheet with given name already exists

    #work out row and column offset of the cell we want to import to
    gsheet_range_offset = a1_to_rowcol(import_to_range)
    gsheet_row_offset = int(gsheet_range_offset[0]) - 1
    gsheet_col_offset = int(gsheet_range_offset[1]) - 1

    #add our offsets to the size of the df so we can work out the cell reference of the end of the range we are importing to
    last_col = len(df.columns) + gsheet_col_offset
    last_row = df.shape[0] + gsheet_row_offset
    gsheet_range_end_a1_ref = rowcol_to_a1(last_row, last_col)

    #concat sheet name & the cell references of the start + end of the range to get the full range reference (in A1 notation)
    gsheet_range = gsheet_tab_name + '!' + import_to_range + ':' + str(gsheet_range_end_a1_ref)

    write_df_to_sheet(df, gsheet_id, gsheet_range)


def rowcol_to_a1(row, column):
    """
    Converts row and column addresses to A1 notation. (e.g. (5, 4) becomes 'D5')

    Args:
        row (int): Row number to be converted to A1 Notation.
        column (int): Column number to be converted to A1 Notation.
    
    Returns:
        cell_ref (str): cell reference in A1 notation.
    """
    #same logic as https://github.com/burnash/gspread/blob/master/gspread/utils.py with slight adjustments
    row = int(row)
    column = int(column)

    if row < 1 or column < 1:
        raise ValueError('row and column ref must be 1 or greater')

    div = column
    column_ref = ''

    while div:
        (div, mod) = divmod(div, 26)
        if mod == 0:
            mod = 26
            div -= 1
        column_ref = chr(mod + 64) + column_ref
    
    cell_ref = str(column_ref) + str(row)
    return cell_ref


def a1_to_rowcol(a1_cell_ref):
    """
    Converts A1 notation cell reference into row & column numbers as integers in a tuple.

    Args:
        a1_cell_ref (str): cell reference in A1 notation (e.g. 'A1', 'D5', 'AQ34' etc)
    
    Returns:
        row_col_ref (tuple): tuple of the row and column numbers as integers, indexed from 1 (e.g. 'A1'→(1, 1), 'Q4'→(4, 17), etc).
    """
    A1_ADDR_ROW_COL_RE = re.compile(r"([A-Za-z]+)?([1-9]\d*)?$")
    m = A1_ADDR_ROW_COL_RE.match(a1_cell_ref)
    if m:
        column_label, row = m.groups()

        if column_label:
            col = 0
            for i, c in enumerate(reversed(column_label.upper())):
                col += (ord(c) - 64) * (26**i)
        else:
            col = None

        if row:
            row = int(row)
        else:
            row = None
    else:
        raise ValueError(str(a1_cell_ref) + ' could not be parsed as an A1 cell reference')

    return (row, col)


def concat_col_headers_to_df(df):
    """
    Concatenates the column names of a dataframe to the beginning of the dataframe.
    Useful when exporting a df to google sheets/excel and you want to include the column headers.

    Args:
        df (pandas.DataFrame): Dataframe which will have the column names concatenated to itself
    """
    column_headers = df.columns.tolist()
    df_column_headers = pd.DataFrame(column_headers).T
    df_column_headers.columns = df_column_headers.iloc[0]
    df = pd.concat([df_column_headers, df], ignore_index=True)
    return df


def create_new_sheet(spreadsheet_id, new_sheet_name='new sheet', tab_colour=(0.0, 0.0, 0.0)):
    """
    Create new sheet within a Gsheet with given name.
    
    Args:
        spreadsheet_id (str): id of the GSheet to create a new sheet inside.
        new_sheet_name (str): name that the new sheet should have.
        tab_colour (tuple): Tuple containing RGB values to set the tab colour.
    """
    creds = get_gsheet_api_credentials()
    service = build('sheets', 'v4', credentials=creds, cache_discovery=False)
    sheet = service.spreadsheets()
    try:
        request_body = {
            'requests': [{
                'addSheet': {
                    'properties': {
                        'title': new_sheet_name,
                        'tabColor': {
                            'red': tab_colour[0],
                            'green': tab_colour[1],
                            'blue': tab_colour[2]
                        }
                    }
                }
            }]
        }

        sheet.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=request_body
        ).execute()


    except:
        logging.info('couldn\'t create new sheet with name \''+str(new_sheet_name)+'\', it may already exist.')


def write_df_to_sheet(df, spreadsheet_id, range_name, value_input_option='USER_ENTERED'):
    """
    Writes a dataframe to a given Google Sheet using the sheets API.
    
    Args:
        df (pandas.DataFrame): dataframe to be written to the Gsheet.
        speadsheet_id (str): id of the GSheet the dataframe will be written to.
        range_name (str): range of the GSheet where the dataframe will be inserted (A1 format e.g. 'Sheet1'!A1:C50).
        value_input_option (str): affects how the GSheets API will interpret the input, best to leave this as 'USER_ENTERED'. see: https://developers.google.com/sheets/api/reference/rest/v4/ValueInputOption
    """
    df = df.fillna('') # NaN values will create invalid JSON payload so need to be replaced
    _values = df.values.tolist()

    # _values = [['A', 'B'], ['C', 'D']]        # test vals
    # range_name = 'cleaned_daily_data!A1:B2'

        # pylint: disable=maybe-no-member
    try:
        creds = get_gsheet_api_credentials()
        service = build('sheets', 'v4', credentials=creds, cache_discovery=False)
        
        body = {
            'values': _values
        }
        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id, range=range_name,
            valueInputOption=value_input_option, body=body).execute()
        logging.info(str({result.get('updatedCells')})+' cells updated.')
        #print(f"{result.get('updatedCells')} cells updated.")
        return result
    except HttpError as error:
        logging.error(f'An error occurred: {error}')
        #print(f"An error occurred: {error}")
        return error


def num_to_col_ref(value, offset=0):
    """
    Converts column number to alphabetical reference for GSheets/Excel. (e.g. 1→A, 2→B, 26→Z, etc).
    Warning: only works up to 26^2 (675 aka 'YY')
    
    Args:
        value (int): Column number to convert to alphabetical reference (A1 notation).
        offset(int): Optional. Offset to column number (0 by default)
    
    Returns:
        alphabetical reference for column number (e.g. 1→A, 2→B, 26→Z, etc)
    """
    value = value + offset
    alphabet_multiples = trunc(value / 26)
    multi_letter = ''

    if alphabet_multiples > 0:
        multi_letter = chr(ord('@')+alphabet_multiples)
        value = value - 26*alphabet_multiples

    letter = multi_letter + chr(ord('@')+value)
    return letter


def flag_outliers(df_raw_vol, stddev_threshold=3, rolling_window=13):
    """
    flags statistical outliers (stddev method) in dataframe time series data, grouped by business line

    Args:
        df_raw_vol (pandas.DataFrame): Dataframe to be tested for outliers. Column names must be: ['date', 'business_line_alias', 'value'].
        stddev_threshold (int): Number of standard deviations from the mean to be considered an outlier. Default is 3.
        rolling_window (int): size of the rolling window in weeks to calculate stddev and mean. Default is 13.

    Returns:
        Original Dataframe with new columns flagging outliers and allowing further analysis: ['dow', 'dom', 'is_stdv_stats_outlier', 'temp_value', 'rolling_mean', 'rolling_stddev', 'low_outlier_threshold', 'high_outlier_threshold'].
    """
    df_raw_vol['date'] = pd.to_datetime(df_raw_vol['date'], format='%d/%m/%Y')

    # add some columns we need
    df_raw_vol['dow'] = df_raw_vol['date'].dt.day_of_week+1         # +1 so Monday=1 as per ISO 8601
    df_raw_vol['dom'] = df_raw_vol['date'].dt.day
    df_raw_vol['is_stdv_stats_outlier'] = False
    df_raw_vol['temp_value'] = df_raw_vol['value']

    old_outlier_count = 0
    flag_outliers_stddev_method(df_raw_vol, stddev_threshold, rolling_window) #first iteration
    new_outlier_count = df_raw_vol.is_stdv_stats_outlier.sum()

    logging.info('Begin outlier detection:')
    logging.info('    initial outlier count: '+str(df_raw_vol.is_stdv_stats_outlier.sum()))

    # iterate until we stop detecting new outliers
    while old_outlier_count != new_outlier_count:
        flag_outliers_stddev_method(df_raw_vol, stddev_threshold, rolling_window)
        old_outlier_count = new_outlier_count
        new_outlier_count = df_raw_vol.is_stdv_stats_outlier.sum()
        #print("new outlier count: "+str(new_outlier_count))
        logging.info('    new outlier count: '+str(new_outlier_count))

    return df_raw_vol


def flag_outliers_stddev_method(df_raw_vol, stddev_threshold, rolling_window):
    """
    Flags statistical outliers using stddev method and a rolling window, grouped by dow & business line
    
    Args:
        df_raw_vol (pandas.DataFrame): Dataframe containing raw volume data to be checked for outliers.
        stddev_threshold (numeric): Threshold at which to consider a value an outlier if it is more than this number of stddevs from the mean.
        rolling_window (int): size of the rolling window to calculate rolling mean and sttdev over.
    
    Returns:
        pandas.DataFrame: Copy of df_raw_vol containing new columns with outlier threshold, rolling mean, and outliers flagged.
    """
    df_raw_vol['rolling_stddev'] = df_raw_vol.groupby(['dow', 'business_line_alias'])['temp_value'].transform(lambda x: x.rolling(window=rolling_window, center=True, min_periods=1).std())
    df_raw_vol['rolling_mean'] = df_raw_vol.groupby(['dow', 'business_line_alias'])['temp_value'].transform(lambda x: x.rolling(window=rolling_window,  center=True, min_periods=1).mean())

    df_raw_vol['low_outlier_threshold'] = df_raw_vol['rolling_mean']-(df_raw_vol['rolling_stddev'] * stddev_threshold)
    df_raw_vol['high_outlier_threshold'] = df_raw_vol['rolling_mean']+(df_raw_vol['rolling_stddev'] * stddev_threshold)

    df_raw_vol.loc[((df_raw_vol['low_outlier_threshold'] > df_raw_vol['temp_value']) & (df_raw_vol['rolling_mean'] - df_raw_vol['temp_value'] > 20)) | ((df_raw_vol['high_outlier_threshold'] <= df_raw_vol['temp_value']) & (df_raw_vol['temp_value'] - df_raw_vol['rolling_mean'] > 20)), 'is_stdv_stats_outlier'] = True
    df_raw_vol.loc[df_raw_vol['is_stdv_stats_outlier'] == True, 'temp_value'] = float("NaN")    # set outliers temp_value to NaN so they are excluded from the calculations of subsequent iterations.

    return df_raw_vol
    

def flag_holidays(df_raw_vol, df_holidays):
    """
    Flag holidays in a dataframe by merging with a df containing holidays.
    
    Args:
        df_raw_vol (pandas.DataFrame): df containining daily volume data. Must have the columns: date, business_line_alias.
        df_holidays (pandas.DataFrame): df containing list of holidays for each business line, merged on to df_raw_vol.
    
    Returns:
        pandas.DataFrame: Copy of df_raw_vol with extra columns flagging holidays for each business line.
    """
    df_holidays['date'] = pd.to_datetime(df_holidays['date'], format='%d/%m/%Y')

    df_raw_vol = df_raw_vol.merge(df_holidays, left_on=['date', 'business_line_alias'], right_on=['date', 'business_line_alias'], how='left')

    df_raw_vol['is_holiday'] = False
    df_raw_vol.loc[df_raw_vol['holiday_name'].notna(), 'is_holiday'] = True

    return df_raw_vol


def import_gsheet_to_df(input_sheet_id, input_sheet_range, has_header_row=True):
    """
    Import a range from a Gsheet and return it as a dataframe.
    
    Args:
        input_sheet_id (str): ID of Google Sheet to import from.
        input_sheet_range (str): Range to import from the GSheet (in A1 format e.g. 'raw'!A1:C).
        has_header_row (bool): If true then rename the df column labels with the header row of the data, then drop it from the df. True by default.

    Returns:
        pandas.DataFrame: Dataframe containing imported range.
    """
    try:
        logging.info( 'Importing range: \'' + input_sheet_range + '\' from G-sheet with id: \'' + input_sheet_id + '\'')
        creds = get_gsheet_api_credentials()
        service = build('sheets', 'v4', credentials=creds, cache_discovery=False)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=input_sheet_id,
                                    range=input_sheet_range).execute()
        values = result.get('values', [])

        if not values:
            logging.warning('No data found in range: \'' + input_sheet_range + '\' from G-sheet id: ' + input_sheet_id + '\'')
            return

        df = pd.DataFrame(values)

        #if the data has a header row, rename the df columns with first row, then drop it from the data
        if has_header_row:
            df.columns = df.iloc[0]
            df = df.drop([0]) 
        
        return df

    except HttpError as err:
        logging.error(err)


def get_gsheet_api_credentials():
    """
    Get credentials for Google Sheets API.
    Checks if the user has valid credentials stored (token.json), or otherwise ask the user to auth via Google SSO.

    Returns:
        (obj): Credentials for Google Sheets API
    """
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets'] # If scope is modified then token.json must be deleted
    creds = None

    # The file token.json stores the user's access and refresh tokens, and is created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        logging.info('    G-Sheets API Credentials: token.json exists')
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logging.info('    User credentials expired, refreshing')
            creds.refresh(Request())
        else:
            logging.info('    Creating new token.json')
            flow = InstalledAppFlow.from_client_secrets_file(
                SECRET_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    ## Authentication end

    return creds


def get_intra_month_seasonality(df_raw_vol, value_col='temp_value', window_size=14, wrap_spline=True):
    """
    Calculates the intra-month seasonality from a provided dataframe using rolling avg method with a smoothing spline & returns it as a new dataframe.
    The df must contain columns called 'business_line_alias' (the operation will be grouped by this column), 'date', 'dom' (day of month) and whichever column contains the value.

    Args:
        df_raw_vol (pandas.DataFrame): Dataframe containing raw data to calculate the intra-month seasonality from.
        value_col (str): The name of the df column which contains the values to calculate the seasonality of.
        window_size (int): The size of the window to calculate seasonality over, default is 14.
        wrap_spline (bool): If true then when the seasonality is smoothed by the spline function it will 'wrap' the data so it cycles smoothly.
    
    Returns:
        pandas.Dataframe: Dataframe containing the intra-month seasonality.
    """

    logging.info('get_intra_month_seasonality: window size: ' + str(window_size))
    #df_raw_vol['temp_value'] = df_raw_vol['value']
    #df_raw_vol.loc[df_raw_vol['exclude_from_seasonality_calcs_flg'] == True, 'temp_value'] = df_raw_vol['rolling_mean']
    #print(df_raw_vol.value_col.dtype())

    # filter raw vols with window size
    df_raw_vol['date'] = pd.to_datetime(df_raw_vol['date']) # date format (change)
    max_date = max(df_raw_vol['date'])
    window_size = window_size * 31 # multiply by 31 to get window size in days
    min_date = max_date - timedelta(days = window_size)
    df_raw_vol = df_raw_vol[(df_raw_vol['date'] >= min_date)]

    df_raw_vol = df_raw_vol.groupby(['dom', 'business_line_alias']).mean(numeric_only=True) #numeric only is added (change)
    #print(df_intra_month_seasonality)
    df_raw_vol['seasonality'] = df_raw_vol.groupby('business_line_alias')[value_col].transform(lambda x: (x - x.mean()) / x.mean()) #instead of apply, transform (change)

    df_raw_vol = df_raw_vol.reset_index() # so that the index columns become part of the dataframe data again (otherwise they won't be exported)
    print('raw seasonality null check: ' + str(df_raw_vol['seasonality'].isna().sum()))
    ss_mean = df_raw_vol['seasonality'].mean()
    df_raw_vol['seasonality'].fillna(value = ss_mean, inplace = True) # sometimes there can be empty values in the seasonality column, just fill them with the mean. TODO : improve this! Work out why NaNs appear and stop them getting to this step
    fitted_spline_df = spline_fitting_grouped(df_raw_vol, 'dom', 'seasonality', 'business_line_alias', wrap_spline)
    #print(fitted_spline_df)
    
    fitted_spline_df.rename(columns={'seasonality':'intra_month_seasonality'},inplace=True) #(change)

    return fitted_spline_df


def get_intra_week_distro(df_raw_vol, window_size = 16):
    """
    Calculates the intra-week split from a provided dataframe & returns it as a new dataframe.

    Args:
        df_raw_vol (pandas.DataFrame): Dataframe containing raw data to calculate the intra-week distribution from.
        window_size (int): Size of the window (in weeks) to calculate the intra-week distribution over.
    
    Returns:
        pandas.Dataframe: Dataframe containing the intra-week distrubution for each grouping.
    """
    logging.info('get_intra_week_distro: window_size: ' + str(window_size))
    df_raw_vol['temp_value'] = df_raw_vol['value']
    df_raw_vol['temp_value'] = pd.to_numeric(df_raw_vol['temp_value'], downcast="float") #tonum
    df_raw_vol.loc[df_raw_vol['exclude_from_seasonality_calcs_flg'] == True, 'temp_value'] = df_raw_vol['rolling_mean']
    df_raw_vol['temp_value'] = pd.to_numeric(df_raw_vol['temp_value'], downcast="float") #tonum

    #filter dates
    df_raw_vol['date'] = pd.to_datetime(df_raw_vol['date']) # date format (change)
    max_date = max(df_raw_vol['date'])
    window_size = window_size * 7 # multiply by 7 to get window size in days
    min_date = max_date - timedelta(days = window_size)
    df_raw_vol = df_raw_vol[(df_raw_vol['date'] >= min_date)]

    df_raw_vol = df_raw_vol.groupby(['dow', 'business_line_alias']).mean(numeric_only=True) #numeric only is added (change)
    #print(df_intra_month_seasonality)
    df_raw_vol['seasonality'] = df_raw_vol.groupby('business_line_alias')['temp_value'].transform(lambda x: (x - x.mean()) / x.mean()) # instead of apply transform  (change)
    df_raw_vol['intra_week_distro'] = df_raw_vol['temp_value'] / df_raw_vol.groupby('business_line_alias')['temp_value'].transform('sum')

    df_raw_vol = df_raw_vol.reset_index() # so that the index columns become part of the dataframe data again (otherwise they won't be exported)
    df_raw_vol['avg_vol'] = df_raw_vol['temp_value']
    df_raw_vol = df_raw_vol.drop(['value', 'dom', 'is_stdv_stats_outlier', 'temp_value', 'rolling_stddev', 'rolling_mean', 'low_outlier_threshold', 'high_outlier_threshold', 'is_holiday', 'exclude_from_seasonality_calcs_flg'], axis=1)

    return df_raw_vol


#wip
def prep_df_for_gsheet_export():
    """prepares dataframe for export to Gsheets by putting indexes back into columns, removing nulls & turning dates into strings"""


def spline_fitting_grouped(df, x_column, y_column, group_column, wrap_spline=True):
    """
    Perform spline fitting on a given column from a dataframe, grouped by a given grouping column.

    Args:
        df (pandas.DataFrame): Dataframe containing the data to be fitted.
        x_column (str): Name of the column containing the x-axis data.
        y_column (str): Name of the column containing the y-axis data.
        group_column (str): Name of the column containing the grouping data.
        wrap_spline (bool): Optional. Should the spline wrap around the data to reduce possibility of a big delta between start & end of spline? True by default.

    Returns:
        pandas.DataFrame: Dataframe containing the fitted spline values.
    """

    # Get grouped data
    grouped_data = df.groupby(group_column)

    # Fit spline for each group
    spline_fits = []
    for z_val, group_data in grouped_data:
               
        x = group_data[x_column]
        y = group_data[y_column]

        # estimates good smoothing value
        s = _estimate_s(x, y)
        #logging.info('    spline fit: ' + z_val + ' estimated s value: ' + str(round(s, 3)))

        # if wrap_spline is true then duplicate the x & y data
        if wrap_spline == True:
            x_len = len(x)
            x_lower_dupe = x.copy()
            x_lower_dupe = x_lower_dupe.apply(lambda x: x - x_len)
            x_upper_dupe = x.copy()
            x_upper_dupe = x_upper_dupe.apply(lambda x: x + x_len)
            x = pd.concat([x_lower_dupe, x, x_upper_dupe])
            y = pd.concat([y.copy(), y, y.copy()])

        spline = UnivariateSpline(x, y, s=s)

        # Get spline values
        spline_values = spline(x)

        # Create new dataframe with spline values
        grouped_spline_values = pd.DataFrame({x_column: x, y_column: spline_values})

        # Add z_column to grouped spline values
        grouped_spline_values[group_column] = z_val
        grouped_spline_values['avg_vol'] = y

        spline_fits.append(grouped_spline_values)

    # Combine fitted spline values from each group
    fitted_spline_values = pd.concat(spline_fits)

    # remove any dom values outside of the 1-31 range (they can exist if wrap_spline is true)
    fitted_spline_values = fitted_spline_values[(fitted_spline_values['dom'] >= 1) & (fitted_spline_values['dom'] <= 31)]

    return fitted_spline_values


def _estimate_s(x, y):
    """
    Estimate good s (smoothing) value for scipy univariate spline function.
    Based on the assumption that we want the stdev of the spline fit residuals to be equal to the noise (stdev) of the y data.
    
    Args:
        x (pandas.Series): Series of x-axis data.
        y (pandas.Series): Series of y-axis data.
    
    Returns:
        float: Estimated s (smoothing) value.
    """
    estimated_s = x.count() * y.std()**2
    if estimated_s >= 0.0:
        return estimated_s
    return 0.0


if __name__ == '__main__':
    main('1fvvfhQdJMnLR3tWoUjATybnrVcS5h-5RibIH-61u_Mg', 'raw!A1:C')