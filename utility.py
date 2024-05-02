import pandas as pd

def absolute_change(data: pd.Series, periods: int|float = 1) -> pd.Series:
    """Calculate an n-periods absolute change."""
    return data - data.shift(periods)

def percent_change(data: pd.Series, periods: int|float = 1) -> pd.Series:
    """Calculate an n-periods percentage change."""
    return (data / data.shift(periods) - 1) * 100

def create_index_series(series: pd.Series, base_date: str) -> pd.Series:
    """ Return a series with values adjusted by the index."""
    
    # Check if base_date is in the series index
    if base_date not in series.index:
        raise ValueError("base_date not found in the series index")

    # Calculate the index factor
    base_value = series.loc[base_date]
    if base_value == 0:
        raise ValueError("base_value at base_date is zero, cannot create index")

    indexed_series = (series / base_value) * 100
    return indexed_series

def custom_quarter_to_timestamp(index):
    """
    Convert a pandas PeriodIndex with quarterly periods (custom ending in FEB, MAY, AUG, NOV)
    to a DatetimeIndex where each quarter is represented by the actual end date.

    Parameters:
    - index: PeriodIndex in the format YYYYQX (where X is 1, 2, 3, 4)
    
    Returns:
    - A DatetimeIndex with correct end dates based on custom quarterly periods
    """
    month_mapper = {1: 2, 2: 5, 3: 8, 4: 11}  # Mapping quarter number to ending month
    dates = []

    for period in index:
        year = period.year
        quarter = period.quarter
        end_month = month_mapper[quarter]
        end_date = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
        dates.append(end_date)

    return pd.DatetimeIndex(dates)

def clean_name(name: str) -> str:
    """
    Cleans the series name by removing any extra spaces or characters and capitalizing each word.

    Arguments:
    - name: str - series name to be cleaned.

    Returns:
    - str: cleaned series name.
    """
    # Remove any extra spaces or characters and capitalize each word
    return name.strip().title()

def generate_summary(series_list, ref_date1, ref_date2):
    """
    Generates summary statistics for a list of series at specified reference dates.

    Arguments:
    - series_list: list - list of series for which summary statistics are to be generated.
    - ref_date1: str - first reference date in 'YYYY-MM-DD' format.
    - ref_date2: str - second reference date in 'YYYY-MM-DD' format.

    Returns:
    - DataFrame containing summary statistics for the given series at the specified reference dates.
    """
    # Convert ref_date1 and ref_date2 to Pandas Timestamps
    ref_date1 = pd.to_datetime(ref_date1)
    ref_date2 = pd.to_datetime(ref_date2)

    # Dictionary to store summary statistics
    summary = {}

    # Iterate over each series in the list
    for series in series_list:
        col_name = clean_name(series.name)

        # Calculate variations for series that do not have 'rate' in their name
        if 'Rate' not in col_name:
            yoy_change = percent_change(series, periods=12)
            mom_change = percent_change(series, periods=1)
        else:
            yoy_change = None
            mom_change = None

        yearly_change = absolute_change(series, periods=12)
        monthly_change = absolute_change(series, periods=1)

        # Iterate over each reference date
        for ref_date in [ref_date1, ref_date2]:
            date_label = ref_date.strftime('%b/%y')

            # Populate the summary dictionary with statistics for the current series and reference date
            summary[f'Value - {date_label}'] = summary.get(f'Value - {date_label}', {})
            summary[f'Value - {date_label}'][col_name] = round(series.loc[ref_date], 2)
            summary[f'YoY (%) - {date_label}'] = summary.get(f'YoY (%) - {date_label}', {})
            summary[f'YoY (%) - {date_label}'][col_name] = round(yoy_change.loc[ref_date], 2) if yoy_change is not None else None
            summary[f'MoM (%) - {date_label}'] = summary.get(f'MoM (%) - {date_label}', {})
            summary[f'MoM (%) - {date_label}'][col_name] = round(mom_change.loc[ref_date], 2) if mom_change is not None else None
            summary[f'Yearly Change - {date_label}'] = summary.get(f'Yearly Change - {date_label}', {})
            summary[f'Yearly Change - {date_label}'][col_name] = round(yearly_change.loc[ref_date], 2)
            summary[f'Monthly Change - {date_label}'] = summary.get(f'Monthly Change - {date_label}', {})
            summary[f'Monthly Change - {date_label}'][col_name] = round(monthly_change.loc[ref_date], 2)

    # Convert the summary dictionary to a DataFrame and return
    return pd.DataFrame(summary)