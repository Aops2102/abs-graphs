"""This module does three things:
A. It obtains the freshest time-series data from the 
   Australian Bureau of Statistics (ABS).
B. It allows that data to be searched for a specific series.
C. It provides a short-hand way to plot the ABS data.

In respect of getting data from the ABS, the general 
approach is to:
1. Download the "latest-release" webpage from the ABS.

2. Scan that webpage to find the link(s) to the download
   all-tables zip-file. We do this because the name of
   the file location on the ABS server changes from
   month to month, and varies beyween ABS webpages.

3. Get the URL headers for this file, amd compare freshness
   with the version in the local cache directory (if any).

4. Use either the zip-file from the cache, or download
   a zip-file from the ABS, save it to the cache,
   and use that file.

5. Open the zip-file, and extract each table to a pandas
   DataFrame with a PeriodIndex. And save the metadata
   to a pandas DataFrame. Return all of these DataFrames
   in a dictionary.

Useful information from the ABS website ...
i.   ABS Catalog numbers:
https://www.abs.gov.au/about/data-services/help/abs-time-series-directory

ii.  ABS Landing Pages:
https://www.abs.gov.au/welcome-new-abs-website#navigating-our-web-address-structure."""

# === imports
# standard library imports
import calendar
import io
import re
import zipfile
from collections import namedtuple
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final, TypeVar, TypeAlias, cast

# analytical imports
import pandas as pd
from pandas import Series, DataFrame
from bs4 import BeautifulSoup

# local imports
import common

# === typing information
# public
# an unexpected error when capturing ABS data ...
class AbsCaptureError(Exception):
    """Raised when the data capture process goes awry."""

# abbreviations for columns in the metadata DataFrame
Metacol = namedtuple(
    "Metacol",
    [
        "did",
        "stype",
        "id",
        "start",
        "end",
        "num",
        "unit",
        "dtype",
        "freq",
        "cmonth",
        "table",
        "tdesc",
        "cat",
    ],
)

# An unpacked zipfile and metadata
AbsDict = dict[str, pd.DataFrame]

# keywords to navigate to an ABS landing page
@dataclass(frozen=True)
class AbsLandingPage:
    """Class for identifying ABS landing pages by theme,
    parent-topic and topic."""

    theme: str
    parent_topic: str
    topic: str

@dataclass
class AbsSelectInput:
    """Data used to select muktiple ABS timeseries
    from different sources within the ABS."""

    landing_page: AbsLandingPage
    table: str
    orig_sa: str
    search1: str
    search2: str
    abbr: str
    calc_growth: bool

@dataclass
class AbsSelectOutput:
    """For each series returned, include some useful metadata."""
    series: pd.Series
    cat_id: str
    table: str
    series_id: str
    unit: str
    orig_sa: str
    abbr: str

AbsSelectionDict: TypeAlias = dict[str, AbsSelectInput]
AbsMultiSeries: TypeAlias = dict[str, AbsSelectOutput]
DataT = TypeVar("DataT", Series, DataFrame)  # python 3.11+

# === Constants
SEAS_ADJ: Final[str] = "Seasonally Adjusted"
TREND: Final[str] = "Trend"
ORIG: Final[str] = "Original"

# public
META_DATA: Final[str] = "META_DATA"
metacol = Metacol(
    did="Data Item Description",
    stype="Series Type",
    id="Series ID",
    start="Series Start",
    end="Series End",
    num="No. Obs.",
    unit="Unit",
    dtype="Data Type",
    freq="Freq.",
    cmonth="Collection Month",
    table="Table",
    tdesc="Table Description",
    cat="Catalogue number",
)

# private = constants
_CACHE_DIR: Final[str] = "./ABS_CACHE/"
_CACHE_PATH: Final[Path] = Path(_CACHE_DIR)
_CACHE_PATH.mkdir(parents=True, exist_ok=True)

# === utility functions
# public
def clear_cache() -> None:
    """Clear the cache directory of zip and xlsx files."""

    extensions = ("*.zip", "*.ZIP", "*.xlsx", "*.XLSX")
    for extension in extensions:
        for fs_object in Path(_CACHE_DIR).glob(extension):
            if fs_object.is_file():
                fs_object.unlink()

# === Data capture from the ABS
# private
def _get_abs_page(
        page: AbsLandingPage, 
) -> bytes:
    """Return the HTML for the ABS topic landing page."""

    head = "https://www.abs.gov.au/statistics/"
    tail = "/latest-release"
    url = f"{head}{page.theme}/{page.parent_topic}/{page.topic}{tail}"
    return common.request_get(url)

# private
def _prefix_url(url: str) -> str:
    """Apply ABS URL prefix to relative links."""

    prefix = "https://www.abs.gov.au"
    # remove a prefix if it already exists (just to be sure)
    url = url.replace(prefix, "")
    url = url.replace(prefix.replace("https://", "http://"), "")
    # add the prefix (back) ...
    return f"{prefix}{url}"

# public
@cache
def get_data_links(
    landing_page: AbsLandingPage,
    verbose: bool = False,
    inspect = "",  # for debugging - save the landing page to disk
) -> dict[str, list[str]]:
    """Scan the ABS landing page for links to ZIP files and for
    links to Microsoft Excel files. Return the links in
    a dictionary of lists by file type ending. Ensure relative
    links are fully expanded."""

    # get relevant web-page from ABS website
    page = _get_abs_page(landing_page)

    # save the page to disk for inspection
    if inspect:
        with open(inspect, "w") as file_handle:
            file_handle.write(page.decode("utf-8"))

    # remove those pesky span tags - probably not necessary
    page = re.sub(b"<span[^>]*>", b" ", page)
    page = re.sub(b"</span>", b" ", page)
    page = re.sub(b"\\s+", b" ", page)  # tidy up white space

    # capture all links (of a particular type)
    link_types = (".xlsx", ".zip", ".xls")  # must be lower case
    soup = BeautifulSoup(page, features="lxml")
    link_dict: dict[str, list[str]] = {}
    for link in soup.findAll("a"):
        url = link.get("href")
        if url is None:
            # ignore silly cases
            continue
        for link_type in link_types:
            if url.lower().endswith(link_type):
                if link_type not in link_dict:
                    link_dict[link_type] = []
                link_dict[link_type].append(_prefix_url(url))
                break

    if verbose:
        for link_type, link_list in link_dict.items():
            summary = [x.split("/")[-1] for x in link_list]  # just the file name
            print(f"Found: {len(link_list)} items of type {link_type}: {summary}")

    return link_dict

# private
def _get_abs_zip_file(
    landing_page: AbsLandingPage, 
    zip_table: int, 
    verbose: bool,
    inspect: str,
) -> bytes:
    """Get the latest zip_file of all tables for
    a specified ABS catalogue identifier"""

    link_dict = get_data_links(landing_page, verbose, inspect)

    # happy case - found a .zip URL on the ABS page
    if r".zip" in link_dict and zip_table < len(link_dict[".zip"]):
        url = link_dict[".zip"][zip_table]
        return common.get_file(
            url, _CACHE_PATH, cache_name_prefix=landing_page.topic, verbose=verbose
        )

    # sad case - need to fake up a zip file
    print("A little unexpected: We need to fake up a zip file")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for u in link_dict[".xlsx"]:
            u = _prefix_url(u)
            file_bytes = common.get_file(
                u, _CACHE_PATH, cache_name_prefix=landing_page.topic, verbose=verbose
            )
            name = Path(u).name
            zip_file.writestr(f"/{name}", file_bytes)
    zip_buf.seek(0)
    return zip_buf.read()

# private
def _get_meta_from_excel(
    excel: pd.ExcelFile, tab_num: str, tab_desc: str, cat_id: str
) -> pd.DataFrame:
    """Capture the metadata from the Index sheet of an ABS excel file.
    Returns a DataFrame specific to the current excel file.
    Returning an empty DataFrame, mneans that the meatadata could not
    be identified."""

    # Unfortunately, the header for some of the 3401.0
    #                spreadsheets starts on row 10
    starting_rows = 9, 10
    required = metacol.did, metacol.id, metacol.stype, metacol.unit
    required_set = set(required)
    for header_row in starting_rows:
        file_meta = excel.parse(
            "Index",
            header=header_row,
            parse_dates=True,
            infer_datetime_format=True,
            converters={"Unit": str},
        )
        file_meta = file_meta.iloc[1:-2]  # drop first and last 2
        file_meta = file_meta.dropna(axis="columns", how="all")

        if required_set.issubset(set(file_meta.columns)):
            break

        if header_row == starting_rows[-1]:
            print(f"Could not find metadata for {cat_id}-{tab_num}")
            return pd.DataFrame()

    # make damn sure there are no rogue white spaces
    for col in required:
        file_meta[col] = file_meta[col].str.strip()

    # standarise some units
    file_meta[metacol.unit] = (
        file_meta[metacol.unit]
        .str.replace("000 Hours", "Thousand Hours")
        .replace("$'000,000", "$ Million")
        .replace("$'000", " $ Thousand")
        .replace("000,000", "Millions")
        .replace("000", "Thousands")
    )
    file_meta[metacol.table] = tab_num.strip()
    file_meta[metacol.tdesc] = tab_desc.strip()
    file_meta[metacol.cat] = cat_id.strip()
    return file_meta

# private
def _unpack_excel_into_df(
    excel: pd.ExcelFile, meta: DataFrame, freq: str, verbose: bool
) -> DataFrame:
    """Take an ABS excel file and put all the Data sheets into a single
    pandas DataFrame and return that DataFrame."""

    data = DataFrame()
    data_sheets = [x for x in excel.sheet_names if cast(str, x).startswith("Data")]
    for sheet_name in data_sheets:
        sheet_data = excel.parse(
            sheet_name,
            header=9,
            index_col=0,
        ).dropna(how="all", axis="index")
        data.index = pd.to_datetime(data.index)

        for i in sheet_data.columns:
            if i in data.columns:
                # Remove duplicate Series IDs before merging
                del sheet_data[i]
                continue
            if verbose and sheet_data[i].isna().all():
                # Warn if data series is all NA
                problematic = meta.loc[meta["Series ID"] == i][
                    ["Table", "Data Item Description", "Series Type"]
                ]
                print(f"Warning, this data series is all NA: {i} (details below)")
                print(f"{problematic}\n\n")

        # merge data into a large dataframe
        if len(data) == 0:
            data = sheet_data
        else:
            data = pd.merge(
                left=data,
                right=sheet_data,
                how="outer",
                left_index=True,
                right_index=True,
                suffixes=("", ""),
            )
    if freq:
        if freq in ("Q", "A"):
            month = calendar.month_abbr[
                cast(pd.PeriodIndex, data.index).month.max()].upper()
            freq = f"{freq}-{month}"
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.to_period(freq=freq)

    return data

# regex patterns for the next function
PATTERN_SUBSUB = re.compile(r"_([0-9]+[a-zA-Z]?)_")
PATTERN_NUM_ALPHA = re.compile(r"^([0-9]+[a-zA-Z]?)_[a-zA-z_]+$")
PATTERN_FOUND = re.compile(r"^[0-9]+[a-zA-Z]?$")

# private
def _get_table_name(z_name: str, e_name: str, verbose: bool):
    """Try and get a consistent and unique naming system for the tables
    found in each zip-file. This is a bit fraught because the ABS does
    this differently for various catalog identifiers.
    Arguments:
    z_name - the file name from zip-file.
    e_name - the self reported table name from the excel spreadsheet.
    verbose - provide additional feedback on this step."""

    # first - lets look at table number from the zip-file name
    z_name = (
        z_name.split(".")[0][4:]
        .replace("55003DO0", "")
        .replace("55024", "")
        .replace("55001Table", "")
        .replace("_Table_", "")
        .replace("55001_", "")
        .lstrip("0")
    )

    if result := re.search(PATTERN_SUBSUB, z_name):
        # search looks anywhere in the string
        z_name = result.group(1)
    if result := re.match(PATTERN_NUM_ALPHA, z_name):
        # match looks from the beginning
        z_name = result.group(1)

    # second - lets see if we can get a table name from the excel meta data.
    splat = e_name.replace("-", " ").replace("_", " ").split(".")
    e_name = splat[0].split(" ")[-1].strip().lstrip("0")

    # third - let's pick the best one
    if e_name == z_name:
        r_value = e_name
    elif re.match(PATTERN_FOUND, e_name):
        r_value = e_name
    else:
        r_value = z_name

    if verbose:
        print(f"table names: {z_name=} {e_name=} --> {r_value=}")
    return r_value

# private
def _get_all_dataframes(zip_file: bytes, verbose: bool) -> AbsDict:
    """Get a DataFrame for each table in the zip-file, plus a DataFrame
    for the metadata. Return these in a dictionary
    Arguments:
     - zip_file - ABS zipfile as a bytes array - contains excel spreadsheets
     - verbose - provide additional feedback on this step.
    Returns:
     - either an empty dictionary (failure) or a dictionary containing
       a separate DataFrame for each table in the zip-file,
       plus a DataFrame called META_DATA for the metadata.
    """

    if verbose:
        print("Extracting DataFrames from the zip-file.")
    freq_dict = {"annual": "Y", "biannual": "Q", "quarter": "Q", "month": "M"}
    returnable: dict[str, DataFrame] = {}
    meta = DataFrame()

    with zipfile.ZipFile(io.BytesIO(zip_file)) as zipped:
        for count, element in enumerate(zipped.infolist()):
            # get the zipfile into pandas
            excel = pd.ExcelFile(io.BytesIO(zipped.read(element.filename)))

            # get table information
            if "Index" not in excel.sheet_names:
                print(
                    "Caution: Could not find the 'Index' "
                    f"sheet in {element.filename}. File not included"
                )
                continue

            # get table header information
            header = excel.parse("Index", nrows=8)  # ???
            cat_id = header.iat[3, 1].split(" ")[0].strip()
            table_name = _get_table_name(
                z_name=element.filename,
                e_name=header.iat[4, 1],
                verbose=verbose,
            )
            tab_desc = header.iat[4, 1].split(".", 1)[-1].strip()

            # get the metadata rows
            file_meta = _get_meta_from_excel(excel, table_name, tab_desc, cat_id)
            if len(file_meta) == 0:
                continue

            # establish freq - used for making the index a PeriodIndex
            freqlist = file_meta["Freq."].str.lower().unique()
            if not len(freqlist) == 1 or freqlist[0] not in freq_dict:
                print(f"Unrecognised data frequency {freqlist} for {tab_desc}")
                continue
            freq = freq_dict[freqlist[0]]

            # fix tabulation when ABS uses the same table numbers for data
            # This happens occasionally
            if table_name in returnable:
                tmp = f"{table_name}-{count}"
                if verbose:
                    print(f"Changing duplicate table name from {table_name} to {tmp}.")
                table_name = tmp
                file_meta[metacol.table] = table_name

            # aggregate the meta data
            meta = pd.concat([meta, file_meta])

            # add the table to the returnable dictionary
            returnable[table_name] = _unpack_excel_into_df(
                excel, file_meta, freq, verbose
            )

    returnable[META_DATA] = meta
    return returnable

# public
@cache
def get_abs_data(
    landing_page: AbsLandingPage, 
    zip_table: int = 0, 
    verbose: bool = False,
    inspect: str = "",  # filename for saving the webpage
) -> AbsDict:
    """For the relevant ABS page return a dictionary containing
    a meta-data Data-Frame and one or more DataFrames of actual
    data from the ABS.
    Arguments:
     - page - class ABS_topic_page - desired time_series page in
            the format:
            abs.gov.au/statistics/theme/parent-topic/topic/latest-release
     - zip_table - select the zipfile to return in order as it
            appears on the ABS webpage - default=0
            (e.g. 6291 has four possible tables,
            but most ABS pages only have one).
            Note: a negative zip_file number will cause the
            zip_file not to be recovered and for individual
            excel files to be recovered from the ABS
     - verbose - display additional web-scraping and caching information.
     - inspect - save the webpage to disk for inspection - 
            inspect is the file name."""

    if verbose:
        print(f"In get_abs_data() {zip_table=} {verbose=}")
        print(f"About to get data on: {landing_page.topic.replace('-', ' ').title()} ")
    zip_file = _get_abs_zip_file(landing_page, zip_table, verbose, inspect)
    if not zip_file:
        raise AbsCaptureError("An unexpected empty zipfile.")
    dictionary = _get_all_dataframes(zip_file, verbose=verbose)
    if len(dictionary) <= 1:
        # dictionary should contain meta_data, plus one or more other dataframes
        raise AbsCaptureError("Could not extract dataframes from zipfile")
    return dictionary

# === find ABS data based on search terms
# public
def find_rows(
    meta: DataFrame,
    search_terms: dict[str, str],
    exact: bool = False,
    regex: bool = False,
    verbose: bool = False,
) -> DataFrame:
    """Extract from meta the rows that match the search_terms.
    Arguments:
     - meta - pandas DataFrame of metadata from the ABS
     - search_terms - dictionary - {search_phrase: meta_column_name}
     - exact - bool - whether to match with == or .str.contains()
     - regex - bool - for .str.contains() - use regulare expressions
     - verbose - bool - print additional information when searching.
    Returns a pandas DataFrame (subseted from meta):"""

    meta_select = meta.copy()
    if verbose:
        print(f"In find_rows() {exact=} {regex=} {verbose=}")
        print(f"In find_rows() starting with {len(meta_select)} rows in the meta_data.")

    for phrase, column in search_terms.items():
        if verbose:
            print(f"Searching {len(meta_select)}: term: {phrase} in-column: {column}")

        pick_me = (
            (meta_select[column] == phrase)
            if (exact or column == metacol.table)
            else meta_select[column].str.contains(phrase, regex=regex)
        )
        meta_select = meta_select[pick_me]
        if verbose:
            print(f"In find_rows() have found {len(meta_select)}")

    if verbose:
        print(f"Final selection is {len(meta_select)} rows.")

    if len(meta_select) == 0:
        print("Nothing selected?")

    return meta_select

# === simplified plotting of ABS data ...
# public
def iudts_from_row(row: pd.Series) -> tuple[str, str, str, str, str]:
    """Return a tuple comrising series_id, units, data_description,
    table_number, series_type."""
    return (
        row[metacol.id],
        row[metacol.unit],
        row[metacol.did],
        row[metacol.table],
        row[metacol.stype],
    )

def clean_description(description: str) -> str:
    """Cleans the series description by removing units and extra characters."""
    # Split the description at the first semicolon and remove extra spaces
    cleaned = description.split(';', 1)[0].strip()
    return cleaned

def create_selector_series_dataframe(
        abs_dict: AbsDict, 
        selector: dict[str, str], 
        regex=False, 
        verbose: bool = False) -> pd.DataFrame:
    """Creates a DataFrame with selected series from ABS data.
    
    Arguments:
    - abs_dict: dict[str, DataFrame] - dictionary containing ABS data DataFrames.
    - selector: dict - used with `find_rows()` to select rows from metadata.
    - regex: bool - if True, allows the use of regular expressions in selection.
    - verbose: bool - if True, displays additional information during the process.
    
    Returns:
    - DataFrame containing selected data series, with the series description as column names, cleaned from units and extra characters.
    """
    
    # DataFrame to collect the series
    series_df = pd.DataFrame()

    # Find corresponding rows in metadata using the selector
    selected_rows = find_rows(abs_dict[META_DATA], selector, regex=regex, verbose=verbose)

    # Iterate over each found row to extract the corresponding series
    for _, row in selected_rows.iterrows():
        series_id, _, description, table, _ = iudts_from_row(row)
        # Clean the description to use as column name
        clean_desc = clean_description(description)
        if verbose:
            print(f"Adding series {clean_desc} from table {table}")
        
        # Extract the series and use the cleaned series description as column name
        series_df[clean_desc] = abs_dict[table][series_id]
    
    # Return the DataFrame with all collected series
    return series_df

def compile_series_from_table(
        abs_dict: AbsDict, 
        table_name: str, 
        verbose: bool = False) -> pd.DataFrame:
    """
    Compiles all series from a specified table into a DataFrame.

    Arguments:
    - abs_dict: dict[str, DataFrame] - dictionary containing ABS data and metadata DataFrames.
    - table_name: str - name of the table from which to extract the series.
    - verbose: bool - if True, displays additional information during the process.

    Returns:
    - DataFrame with each series as a column.
    """
    # Check if the specified table is present in the dictionary
    if table_name not in abs_dict:
        raise ValueError("Specified table is not present in the dictionary")

    # DataFrame to collect the series
    series_df = pd.DataFrame()

    # Iterate over each series in the specified DataFrame
    for series_id in abs_dict[table_name].columns:
        # Get the series description from the metadata DataFrame
        if verbose:
            print(f"Adding series ID {series_id} to DataFrame")
        description = abs_dict['META_DATA'].loc[abs_dict['META_DATA']['Series ID'] == series_id, 'Data Item Description'].values[0]

        # Add the series to the resulting DataFrame, using the series description as the column name
        series_df[description] = abs_dict[table_name][series_id]

    return series_df


