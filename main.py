## Python Setup
# Lib imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from datetime import datetime

# Local imports
from abs_data_capture import (
    AbsLandingPage,
    get_abs_data,
    create_selector_series_dataframe,
    compile_series_from_table,
    clear_cache,
    metacol
)

from utility import (
    absolute_change,
    percent_change,
    create_index_series,
    custom_quarter_to_timestamp,
    generate_summary  
)

# Clearing cache data from 
clear_cache()

## Get Data from ABS - Defining landing pages for different ABS (Australian Bureau of Statistics) topics
# Landing page for Labour Force statistics
LANDING_PAGE_LABOUR_FORCE = AbsLandingPage(
    theme="labour",  # Theme of the landing page
    parent_topic="employment-and-unemployment",  # Parent topic of the landing page
    topic="labour-force-australia",  # Specific topic of the landing page
)

# Getting data from the ABS for the Labour Force statistics landing page
abs_dict_lf = get_abs_data(LANDING_PAGE_LABOUR_FORCE)

# Landing page for Job Vacancies statistics
LANDING_PAGE_JOB_VACANCIES = AbsLandingPage(
    theme="labour",  # Theme of the landing page
    parent_topic="employment-and-unemployment",  # Parent topic of the landing page
    topic="job-vacancies-australia",  # Specific topic of the landing page
)

# Getting data from the ABS for the Job Vacancies statistics landing page
abs_dict_jv = get_abs_data(LANDING_PAGE_JOB_VACANCIES)

# Landing page for Jobs statistics
LANDING_PAGE_JOBS = AbsLandingPage(
    theme="labour",  # Theme of the landing page
    parent_topic="labour-accounts",  # Parent topic of the landing page
    topic="labour-account-australia"  # Specific topic of the landing page
)

# Getting data from the ABS for the Jobs statistics landing page
abs_dict_jobs = get_abs_data(LANDING_PAGE_JOBS)

# Landing page for Wage Price Index statistics
LANDING_PAGE_WPI = AbsLandingPage(
    theme="economy",  # Theme of the landing page
    parent_topic="price-indexes-and-inflation",  # Parent topic of the landing page
    topic="wage-price-index-australia",  # Specific topic of the landing page
)

# Getting data from the ABS for the Wage Price Index statistics landing page
abs_dict_wpi = get_abs_data(LANDING_PAGE_WPI)

## Getting Dataframes with ABS Info
# Define dictionaries for different data selectors related to various statistics

# Data selector for trend Labour Force statistics
labour_trend = {
    "1": metacol.table,  # Selector for table type
    "Persons": metacol.did,  # Selector for persons
    "Trend": metacol.stype  # Selector for trend type
}

# Data selector for seasonally adjusted Labour Force statistics
labour_sa = {
    "1": metacol.table,  # Selector for table type
    "Persons": metacol.did,  # Selector for persons
    "Seasonally Adjusted": metacol.stype  # Selector for seasonally adjusted type
}

# Data selector for seasonally adjusted Job Vacancy statistics
job_vacancy_sa = {
    "1": metacol.table,  # Selector for table type
    "Seasonally Adjusted": metacol.stype  # Selector for seasonally adjusted type
}

# Data selector for Job Vacancy by Industry statistics
job_vacancy_by_industry = {
    "4": metacol.table,  # Selector for table type
    "": metacol.did,  # Selector for empty field
    "STOCK": metacol.dtype  # Selector for data type
}

# Data selector for trend Underemployment statistics
underemployment_trend = {
    "22": metacol.table,  # Selector for table type
    "A85256589V": metacol.id,  # Selector for unique identifier
}

# Data selector for seasonally adjusted Underemployment statistics
underemployment_sa = {
    "22": metacol.table,  # Selector for table type
    "A85255725J": metacol.id,  # Selector for unique identifier
}

# Data selector for Jobs Vacancy Rate statistics
jobs_vacancy_rate = {
    "1": metacol.table,  # Selector for table type
    "A85389541V": metacol.id,  # Selector for unique identifier
}

# Data selector for Wage Price Index excluding bonuses Year-over-Year statistics
wpi_ex_bonuses_yoy = {
    "1": metacol.table,  # Selector for table type
    "A83895396W": metacol.id,  # Selector for unique identifier
}

# Data selector for seasonally adjusted Hours Worked statistics
hours_worked_sa = {
    "19": metacol.table,  # Selector for table type
    "Persons": metacol.did,  # Selector for persons
    "Seasonally Adjusted": metacol.stype,  # Selector for seasonally adjusted type
}

# Data selector for trend Hours Worked statistics
hours_worked_trend = {
    "19": metacol.table,  # Selector for table type
    "Persons": metacol.did,  # Selector for persons
    "Trend": metacol.stype,  # Selector for trend type
}

# Create DataFrame objects for each statistic using data from ABS
# DataFrame for trend Labour Force statistics
df_labour_trend = create_selector_series_dataframe(abs_dict_lf, labour_trend)

# DataFrame for seasonally adjusted Labour Force statistics
df_labour_sa = create_selector_series_dataframe(abs_dict_lf, labour_sa)

# DataFrame for seasonally adjusted Job Vacancy statistics
df_job_vacancy_sa = create_selector_series_dataframe(abs_dict_jv, job_vacancy_sa)

# DataFrame for Job Vacancy by Industry statistics
df_job_vacancy_by_industry = compile_series_from_table(abs_dict_jv, "4")

# DataFrame for trend Underemployment statistics
df_under_trend = create_selector_series_dataframe(abs_dict_lf, underemployment_trend)

# DataFrame for seasonally adjusted Underemployment statistics
df_under_sa = create_selector_series_dataframe(abs_dict_lf, underemployment_sa)

# DataFrame for Jobs Vacancy Rate statistics
df_jv_rate_sa = create_selector_series_dataframe(abs_dict_jobs, jobs_vacancy_rate)

# DataFrame for Wage Price Index excluding bonuses Year-over-Year statistics
df_wpi_ex_bonus_yoy_sa = create_selector_series_dataframe(abs_dict_wpi, wpi_ex_bonuses_yoy)

# DataFrame for seasonally adjusted Hours Worked statistics
df_hour_worked_sa = create_selector_series_dataframe(abs_dict_lf, hours_worked_sa)

# DataFrame for trend Hours Worked statistics
df_hour_worked_trend = create_selector_series_dataframe(abs_dict_lf, hours_worked_trend)

# Convert index to timestamp for each DataFrame
df_labour_trend.index = df_labour_trend.index.to_timestamp()
df_labour_sa.index = df_labour_sa.index.to_timestamp()
df_under_trend.index = df_under_trend.index.to_timestamp()
df_under_sa.index = df_under_sa.index.to_timestamp()
df_job_vacancy_sa.index = custom_quarter_to_timestamp(df_job_vacancy_sa.index)
df_jv_rate_sa.index = df_jv_rate_sa.index.to_timestamp()
df_wpi_ex_bonus_yoy_sa.index = df_wpi_ex_bonus_yoy_sa.index.to_timestamp()
df_hour_worked_sa.index = df_hour_worked_sa.index.to_timestamp()
df_hour_worked_trend.index = df_hour_worked_trend.index.to_timestamp()

## Getting Tables and Plots for PDF
# Use a relative path for the PDF
pdf_path = 'Labour Market Report (ABS).pdf'
pdf = PdfPages(pdf_path)

# Define reference dates
ref_date1 = '2024-03-01'
ref_date2 = '2024-02-01'

# Create a list of example series from different DataFrames
series_list_sa = [
    df_labour_sa['Employed total'].rename("Employed total (Thousands)"),
    df_labour_sa['> Employed full-time'].rename("Employed full-time (Thousands)"),
    df_labour_sa['> Employed part-time'].rename("Employed part-time (Thousands)"),
    df_labour_sa['Unemployed total'].rename("Unemployed total (Thousands)"),
    df_labour_sa['Unemployment rate'].rename("Unemployment rate (%)"),
    df_under_sa['Underemployment rate (proportion of labour force)'].rename("Underemployment rate (%)"),
    (df_under_sa['Underemployment rate (proportion of labour force)'] + df_labour_sa['Unemployment rate']).rename("Underutilisation rate (%)"),
    df_labour_sa['Participation rate'].rename("Participation rate (%)"),
    df_hour_worked_sa['Monthly hours worked in all jobs'].rename('Monthly Hours Worked')
]

series_list_nsa = [
    df_labour_trend['Employed total'].rename("Employed total (Thousands)"),
    df_labour_trend['> Employed full-time'].rename("Employed full-time (Thousands)"),
    df_labour_trend['> Employed part-time'].rename("Employed part-time (Thousands)"),
    df_labour_trend['Unemployed total'].rename("Unemployed total (Thousands)"),
    df_labour_trend['Unemployment rate'].rename("Unemployment rate (%)"),
    df_under_trend['Underemployment rate (proportion of labour force)'].rename("Underemployment rate (%)"),
    (df_under_trend['Underemployment rate (proportion of labour force)'] + df_labour_sa['Unemployment rate']).rename("Underutilisation rate (%)"),
    df_labour_trend['Participation rate'].rename("Participation rate (%)"),
]

# Generate the summary table
summary_table_sa = generate_summary(series_list_sa, ref_date1, ref_date2)
summary_table_nsa = generate_summary(series_list_nsa, ref_date1, ref_date2)

# Table 1
# DataFrame
df = summary_table_sa

# Adding indices as the first column in the cell
cell_data = [[str(index)] + list(row) for index, row in zip(summary_table_sa.index, df.values)]

# Adding headers and indices in bold
headers = ['Index'] + [header.replace(' - ', '\n') for header in df.columns.tolist()]
cell_data.insert(0, headers)

# Configuring matplotlib to draw a table
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

fig.suptitle('Table 1', fontsize=16, fontweight='bold', y=0.95)

# Creating the table
table = ax.table(cellText=cell_data, colLabels=None, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

# Configuring table styles
table.auto_set_font_size(True)
table.set_fontsize(11)

# Adjusting column widths
for i, width in enumerate([0.2] + [0.11] * (len(headers) - 1)):
    for j in range(len(cell_data)):
        cell = table[j, i]
        cell.set_width(width)
        cell.set_fontsize(8)

# Setting background color for headers
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#a9a9a9')  # dark gray color

# Alternating colors for data rows
for i in range(1, len(cell_data)):
    color = '#f0f0f0' if i % 2 == 0 else '#ffffff'  # light gray and white colors alternatively
    for j in range(len(headers)):
        table.get_celld()[(i, j)].set_facecolor(color)

# Adicionando título ao gráfico
ax.set_title('Key Statistics - SA')

# Adjusting layout to ensure the plot is centered and fits well within an A4 page
plt.tight_layout(pad=2.0)

# Saving the current figure into the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Table 2
# DataFrame
df = summary_table_nsa

# Adding indices as the first column in the cell
cell_data = [[str(index)] + list(row) for index, row in zip(summary_table_sa.index, df.values)]

# Adding headers and indices in bold
headers = ['Index'] + [header.replace(' - ', '\n') for header in df.columns.tolist()]
cell_data.insert(0, headers)

# Configuring matplotlib to draw a table
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

fig.suptitle('Table 2', fontsize=16, fontweight='bold', y=0.95)

# Creating the table
table = ax.table(cellText=cell_data, colLabels=None, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

# Configuring table styles
table.auto_set_font_size(True)
table.set_fontsize(11)

# Adjusting column widths
for i, width in enumerate([0.2] + [0.11] * (len(headers) - 1)):
    for j in range(len(cell_data)):
        cell = table[j, i]
        cell.set_width(width)
        cell.set_fontsize(8)

# Setting background color for headers
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#a9a9a9')  # dark gray color

# Alternating colors for data rows
for i in range(1, len(cell_data)):
    color = '#f0f0f0' if i % 2 == 0 else '#ffffff'  # light gray and white colors alternatively
    for j in range(len(headers)):
        table.get_celld()[(i, j)].set_facecolor(color)

# Adicionando título ao gráfico
ax.set_title('Key Statistics - NSA')

# Adjusting layout to ensure the plot is centered and fits well within an A4 page
plt.tight_layout(pad=2.0)

# Saving the current figure into the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 1: Labour Market Summary
# Creating the plot with specified dimensions
fig, ax1 = plt.subplots(figsize=(14, 8))

# Adding a page title above the plot
fig.suptitle('Figure 1: Labour Market, Summary', fontsize=16, fontweight='bold', y=0.95)

# Plotting unemployment rate with a red line
ax1.plot(df_labour_sa.index, df_labour_sa['Unemployment rate'], label='Unemployment Rate', color='tab:red')

# Plotting underemployment rate with a blue line
ax1.plot(df_under_sa.index, df_under_sa['Underemployment rate (proportion of labour force)'], label='Underemployment Rate (Proportion of Labour Force)', color='tab:blue')

# Creating a secondary y-axis for the participation rate
ax2 = ax1.twinx()
ax2.plot(df_labour_sa.index, df_labour_sa['Participation rate'], label='Participation Rate (RHS)', color='tab:green', linestyle='--')

# Setting x-axis and y-axis labels
ax1.set_xlabel('Date')
ax1.set_ylabel('(%)')
ax2.set_ylabel('(%)')

# Setting a title for the axes
ax1.set_title('Australia Labour Market')

# Formatting the x-axis to show month/year and adjusting interval to display a tick every six months
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))

# Setting x-axis limits to start from a specific date
ax1.set_xlim([datetime(2009, 10, 1), df_labour_sa.index.max()])

# Adding grid lines to the primary axis
ax1.grid(True, alpha=0.4)

# Rotating x-axis labels for better readability
plt.setp(ax1.get_xticklabels(), rotation=90)

# Adding legends for both primary and secondary y-axes
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Adjusting layout to ensure the plot is centered and fits well within an A4 page
plt.tight_layout(pad=2.0)

# Saving the current figure into the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 2: Labour Force Summary
fig, ax1 = plt.subplots(figsize=(14, 8))

# Add page title
fig.suptitle('Figure 2: Labour Force, Summary', fontsize=16, fontweight='bold', y=0.95)

# Plot total labour force and total employed
ax1.plot(df_labour_sa.index, df_labour_sa['Labour force total'], label='Labour Force Total', color='tab:blue')
ax1.plot(df_labour_sa.index, df_labour_sa['Employed total'], label='Employed Total', color='tab:green')

# Create a second y-axis for total unemployed
ax2 = ax1.twinx()
ax2.plot(df_labour_sa.index, df_labour_sa['Unemployed total'], label='Unemployed Total (RHS)', color='tab:red', linestyle='--')

# Set labels for x and y axes
ax1.set_xlabel('Date')
ax1.set_ylabel('(Thousands)')
ax2.set_ylabel('(Thousands)')

# Set title for the axes
ax1.set_title('Labour Force')

# Format x-axis to show month/year
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))

# Set x-axis limits
ax1.set_xlim([datetime(2009, 10, 1), df_labour_sa.index.max()])

# Add grid to the main axis
ax1.grid(True, alpha=0.4)

# Rotate x-axis labels
plt.setp(ax1.get_xticklabels(), rotation=90)

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Adjust layout to ensure the plot is centered and well-fitted on the A4 page
plt.tight_layout(pad=2.0)

# Save the current figure to the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 3: Employment Growth
fig, ax1 = plt.subplots(figsize=(14, 8))

# Calculating series for plotting
total_employee_growth = percent_change(df_labour_sa['Employed total'], 12)
full_time_employee_growth = percent_change(df_labour_sa['> Employed full-time'], 12)
part_time_employee_growth = percent_change(df_labour_sa['> Employed part-time'], 12)

# Add page title
fig.suptitle('Figure 3: Employment Growth', fontsize=16, fontweight='bold', y=0.95)

# Plot lines
ax1.plot(total_employee_growth.index, total_employee_growth, label='Total', color='tab:red')
ax1.plot(full_time_employee_growth.index, full_time_employee_growth, label='Full-Time', color='tab:blue')
ax1.plot(part_time_employee_growth.index, part_time_employee_growth, label='Part-Time', color='tab:green')

# Set labels for x and y axes
ax1.set_xlabel('Date')
ax1.set_ylabel('(%)')

# Set title for the axes
ax1.set_title('Employment Growth (YoY)')

# Format x-axis to show month/year
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))

# Set x-axis limits
ax1.set_xlim([datetime(2009, 10, 1), total_employee_growth.index.max()])

# Add grid to the main axis
ax1.grid(True, alpha=0.4)

# Rotate x-axis labels
plt.setp(ax1.get_xticklabels(), rotation=90)

# Add legends
ax1.legend(loc='upper left')

# Adjust layout to ensure the plot is centered and well-fitted on the A4 page
plt.tight_layout(pad=2.0)

# Save the current figure to the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 4: Employment Change
fig, ax1 = plt.subplots(figsize=(14, 8))

# Calculating series for plotting
abs_total_employee_growth = absolute_change(df_labour_sa['Employed total'], 1)
abs_full_time_employee_growth = absolute_change(df_labour_sa['> Employed full-time'], 1)
abs_part_time_employee_growth = absolute_change(df_labour_sa['> Employed part-time'], 1)

# Add page title
fig.suptitle('Figure 4: Employment Change', fontsize=16, fontweight='bold', y=0.95)

# Plot lines
ax1.plot(abs_total_employee_growth.index, abs_total_employee_growth, label='Total', color='tab:red')
ax1.bar(abs_full_time_employee_growth.index, abs_full_time_employee_growth, 25, label='Full-Time', color='tab:blue')
ax1.bar(abs_part_time_employee_growth.index, abs_part_time_employee_growth, 25, label='Part-Time', color='tab:green')

# Set labels for x and y axes
ax1.set_xlabel('Date')
ax1.set_ylabel('(Changes)')

# Set title for the axes
ax1.set_title('Employment Changes (MoM)')

# Format x-axis to show month/year
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))

# Set x-axis limits
ax1.set_xlim([datetime(2009, 10, 1), abs_total_employee_growth.index.max()])

# Add grid to the main axis
ax1.grid(True, alpha=0.4)

# Rotate x-axis labels
plt.setp(ax1.get_xticklabels(), rotation=90)

# Add legends
ax1.legend(loc='upper left')

# Adjust layout to ensure the plot is centered and well-fitted on the A4 page
plt.tight_layout(pad=2.0)

# Save the current figure to the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 5: Employment Comparision
fig, ax1 = plt.subplots(figsize=(14, 8))

# Calculating series for plotting
idx_total_employee = create_index_series(df_labour_sa['Employed total'], '2000-01-01')
idx_full_time_employee = create_index_series(df_labour_sa['> Employed full-time'], '2000-01-01')
idx_part_time_employee = create_index_series(df_labour_sa['> Employed part-time'], '2000-01-01')

# Add page title
fig.suptitle('Figure 5: Employment Comparision', fontsize=16, fontweight='bold', y=0.95)

# Plot lines
ax1.plot(idx_total_employee.index, idx_total_employee, label='Total', color='tab:red')
ax1.plot(idx_full_time_employee.index, idx_full_time_employee, label='Full-Time', color='tab:blue')
ax1.plot(idx_part_time_employee.index, idx_part_time_employee, label='Part-Time', color='tab:green')

# Set labels for x and y axes
ax1.set_xlabel('Date')
ax1.set_ylabel('(Index)')

# Set title for the axes
ax1.set_title('Employment Comparision: Level − Jan/00 = 100')

# Format x-axis to show month/year
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))

# Set x-axis limits
ax1.set_xlim([datetime(2009, 10, 1), idx_total_employee.index.max()])

# Calculate the maximum value among the series
max_value = np.max([idx_total_employee.max(), idx_full_time_employee.max(), idx_part_time_employee.max()])

# Set y-axis limits with minimum value as 100 and maximum value as the maximum value among the series
ax1.set_ylim([100, max_value + 10])

# Add grid to the main axis
ax1.grid(True, alpha=0.4)

# Rotate x-axis labels
plt.setp(ax1.get_xticklabels(), rotation=90)

# Add legends
ax1.legend(loc='upper left')

# Adjust layout to ensure the plot is centered and well-fitted on the A4 page
plt.tight_layout(pad=2.0)

# Save the current figure to the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 6: Job Vacancies
fig, ax1 = plt.subplots(figsize=(14, 8))

# Add page title
fig.suptitle('Figure 6: Job Vacancies', fontsize=16, fontweight='bold', y=0.95)

# Plot lines
ax1.plot(df_job_vacancy_sa.index, df_job_vacancy_sa['Job Vacancies'], label='Job Vacancies', color='tab:red')

# Set labels for x and y axes
ax1.set_xlabel('Date')
ax1.set_ylabel('(Thousands)')

# Set title for the axes
ax1.set_title('Job Vancancies')

# Format x-axis to show month/year
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))

# Set x-axis limits
ax1.set_xlim([datetime(2009, 10, 1), df_job_vacancy_sa.index.max()])

# Add grid to the main axis
ax1.grid(True, alpha=0.4)

# Rotate x-axis labels
plt.setp(ax1.get_xticklabels(), rotation=90)

# Add legends
ax1.legend(loc='upper left')

# Adjust layout to ensure the plot is centered and well-fitted on the A4 page
plt.tight_layout(pad=2.0)

# Save the current figure to the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 7: Employment vs. Jobs Vacancies
fig, ax1 = plt.subplots(figsize=(14, 8))

# Add page title
fig.suptitle('Figure 7: Employment vs. Jobs Vacancies', fontsize=16, fontweight='bold', y=0.95)

# Calculating series for plotting
job_vancancy_growth = percent_change(df_job_vacancy_sa['Job Vacancies'], 4)

# Plot total labour force and total employed
ax1.plot(idx_total_employee.index, idx_total_employee, label='Total', color='tab:blue')

# Create a second y-axis for total unemployed
ax2 = ax1.twinx()
ax2.plot(job_vancancy_growth.index, job_vancancy_growth, label='Job Vacancies − YoY (RHS)', color='tab:red', linestyle='--')

# Set labels for x and y axes
ax1.set_xlabel('Date')
ax1.set_ylabel('(Index)')
ax2.set_ylabel('(%)', color='tab:red')

# Set title for the axes
ax1.set_title('Australia − Job Vacancies and Employment')

# Format x-axis to show month/year
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))

# Set x-axis limits
ax1.set_xlim([datetime(2010, 9, 1), idx_total_employee.index.max()])

# Set y-axis limits with minimum value as 100 and maximum value as the maximum value among the series
ax1.set_ylim([100, idx_total_employee.max()+ 10])

# Add grid to the main axis
ax1.grid(True, alpha=0.4)

# Rotate x-axis labels
plt.setp(ax1.get_xticklabels(), rotation=90)

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Adjust layout to ensure the plot is centered and well-fitted on the A4 page
plt.tight_layout(pad=2.0)

# Save the current figure to the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 8: Job Vacancy - By Industry (Covid)
# Filtering relevant columns from the DataFrame to exclude 'Total All Industries' and 'Standard Error' related columns
job_vacancy_columns = [col for col in df_job_vacancy_by_industry.columns if 'Job Vacancies' in col and 'Total All Industries' not in col and 'Standard Error' not in col]

# Selecting data specifically for the first quarters of 2020 and 2024
data_feb20 = df_job_vacancy_by_industry.loc['2020Q1', job_vacancy_columns]
data_feb24 = df_job_vacancy_by_industry.loc['2024Q1', job_vacancy_columns]

# Calculating the percentage change between 2020 and 2024 data
change = (data_feb24 - data_feb20)

# Cleaning labels by removing unnecessary parts and preparing them for the plot
labels_jv_by_industry = [col.split(';')[1].strip() for col in job_vacancy_columns]

# Creating a sorted series from the calculated change data
change_series = pd.Series(data=change.values, index=labels_jv_by_industry)
change_sorted = change_series.sort_values()

# Creating the plot with specified dimensions
fig, ax1 = plt.subplots(figsize=(14, 8))

# Adding a page title above the plot
fig.suptitle('Figure 8: Job Vacancy - By Industry (Covid)', fontsize=16, fontweight='bold', y=0.95)

bars = plt.bar(change_sorted.index, change_sorted.values, color='tab:red')

# Adding value labels on top of each bar for better clarity
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=8)

# Rotating x-axis labels to avoid overlap and improve readability
plt.xticks(rotation=90)

# Setting the y-axis label to indicate the unit of measurement
plt.ylabel('(Thousands)')

# Adding a title to the plot for context
plt.title('Job Vacancy by Industry - Covid Change (Feb/20 to Feb/24)')

# Enabling grid lines on the y-axis for better reference of magnitude
plt.grid(axis='y')

# Adjusting layout to ensure the plot is centered and fits well within an A4 page
plt.tight_layout(pad=2.0)

# Saving the current figure into the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 9: Job Vacancy - By Industry (Quarter)
# Selecting data specifically for the first quarters of 2020 and 2024
data_nov23 = df_job_vacancy_by_industry.loc['2023Q4', job_vacancy_columns]

# Calculating the percentage change between 2020 and 2024 data
change_quarter = (data_feb24 - data_nov23)

# Creating a sorted series from the calculated change data
change_quarter_series = pd.Series(data=change_quarter.values, index=labels_jv_by_industry)
change_quarter_sorted = change_quarter_series.sort_values()

# Creating the plot with specified dimensions
fig, ax1 = plt.subplots(figsize=(14, 8))

# Adding a page title above the plot
fig.suptitle('Figure 9: Job Vacancy - By Industry (Quarter)', fontsize=16, fontweight='bold', y=0.95)

bars = plt.bar(change_quarter_sorted.index, change_quarter_sorted.values, color='tab:red')

# Adding value labels on top of each bar for better clarity
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', fontsize=8)

# Rotating x-axis labels to avoid overlap and improve readability
plt.xticks(rotation=90)

# Setting the y-axis label to indicate the unit of measurement
plt.ylabel('(Thousands)')

# Adding a title to the plot for context
plt.title('Job Vacancy by Industry - Quarter Change (Nov/23 to Feb/24)')

# Enabling grid lines on the y-axis for better reference of magnitude
plt.grid(axis='y', alpha=0.4)

# Adjusting layout to ensure the plot is centered and fits well within an A4 page
plt.tight_layout(pad=2.0)

# Saving the current figure into the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 10: Beveridge Curve
# Preparing series for plot
unemployed_rate_q = df_labour_sa['Unemployment rate'].resample('QS-OCT').mean()

# Preparind Data
aligned_data = pd.DataFrame({'Unemployment Rate': unemployed_rate_q, 'Job Vacancy Rate': df_jv_rate_sa['Jobs']}).dropna()

# Defining periods stamps for plot
periods = {
    'Jan 2001 - Dec 2008': ('2001-01-01', '2008-12-31'),
    'Jan 2009 - Jun 2020': ('2009-01-01', '2020-06-30'),
    'Jul 2020 - Feb 2024': ('2020-07-01', '2024-02-29'),
}

# Creating the plot with specified dimensions
fig, ax = plt.subplots(figsize=(14, 8))

# Adding a page title above the plot
fig.suptitle('Figure 10: Beveridge Curve', fontsize=16, fontweight='bold', y=0.95)

# Ploting data for intervals
for period, (start_date, end_date) in periods.items():
    mask = (aligned_data.index >= start_date) & (aligned_data.index <= end_date)
    ax.plot(aligned_data[mask]['Unemployment Rate'], aligned_data[mask]['Job Vacancy Rate'], marker='o', label=period)

# Adding titles and lables
ax.set_title('Australia, Beveridge Curve (Quarter)')
ax.set_xlabel('Unemployment Rate (%)')
ax.set_ylabel('Job Vacancy Rate (%)')

# Adding grid lines to the primary axis
ax.grid(True, alpha=0.4)

# Adding legends y-axes
ax.legend(loc='upper right')

# Adjusting layout to ensure the plot is centered and fits well within an A4 page
plt.tight_layout(pad=2.0)

# Saving the current figure into the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 11: Unemployment Rate vs Jobs Vacancies
# Creating the plot with specified dimensions
fig, ax1 = plt.subplots(figsize=(14, 8))

# Adding a page title above the plot
fig.suptitle('Figure 11: Unemployment Rate vs Jobs Vacancies', fontsize=16, fontweight='bold', y=0.95)

# Plotting unemployment rate with a red line
ax1.plot(df_labour_sa.index, df_labour_sa['Unemployment rate'], label='Unemployment Rate', color='tab:red')

# Creating a secondary y-axis for the participation rate
ax2 = ax1.twinx()
ax2.plot(df_jv_rate_sa.index, df_jv_rate_sa['Jobs'], label='Job Vacancy Rate (RHS)', color='tab:purple', linestyle='', marker='^')

# Setting x-axis and y-axis labels
ax1.set_xlabel('Date')
ax1.set_ylabel('(%)')
ax2.set_ylabel('(%)')

# Setting a title for the axes
ax1.set_title('Unemployment vs. Job Vacancy')

# Inverting the y-axis
ax2.invert_yaxis()

# Formatting the x-axis to show month/year and adjusting interval to display a tick every six months
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))

# Setting x-axis limits to start from a specific date
ax1.set_xlim([datetime(2009, 10, 1), df_labour_sa.index.max()])

# Adding grid lines to the primary axis
ax1.grid(True, alpha=0.4)

# Rotating x-axis labels for better readability
plt.setp(ax1.get_xticklabels(), rotation=90)

# Adding legends for both primary and secondary y-axes
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Adjusting layout to ensure the plot is centered and fits well within an A4 page
plt.tight_layout(pad=2.0)

# Saving the current figure into the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 12: Unemployment Rate vs. Vacancies to Unemployment
# Defining series for plot
unemployed_total_q = df_labour_sa['Unemployed total'].resample('QS-FEB').sum()
unemployed_total_q.index = custom_quarter_to_timestamp(unemployed_total_q.index)
vacancy_unemployment_rate =  (1 - (df_job_vacancy_sa['Job Vacancies']/(unemployed_total_q + df_job_vacancy_sa['Job Vacancies']))) * 100

# Creating the plot with specified dimensions
fig, ax1 = plt.subplots(figsize=(14, 8))

# Adding a page title above the plot
fig.suptitle('Figure 12: Unemployment Rate vs. Vacancies to Unemployment', fontsize=16, fontweight='bold', y=0.95)

# Plotting unemployment rate with a red line
ax1.plot(df_labour_sa.index, df_labour_sa['Unemployment rate'], label='Unemployment Rate', color='tab:red')

# Creating a secondary y-axis for the participation rate
ax2 = ax1.twinx()
ax2.plot(vacancy_unemployment_rate.index, vacancy_unemployment_rate, label='Vacancy to Unemployment (RHS)', color='tab:purple', linestyle='', marker='^')

# Setting x-axis and y-axis labels
ax1.set_xlabel('Date')
ax1.set_ylabel('(%)')
ax2.set_ylabel('(%)')

# Setting a title for the axes
ax1.set_title('UR vs. Vacancies to Unemployment')

# Formatting the x-axis to show month/year and adjusting interval to display a tick every six months
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))

# Setting x-axis limits to start from a specific date
ax1.set_xlim([datetime(2009, 5, 1), df_labour_sa.index.max()])

# Adding grid lines to the primary axis
ax1.grid(True, alpha=0.4)

# # Inverting the y-axis
# ax2.invert_yaxis()

# Rotating x-axis labels for better readability
plt.setp(ax1.get_xticklabels(), rotation=90)

# Adding legends for both primary and secondary y-axes
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Adjusting layout to ensure the plot is centered and fits well within an A4 page
plt.tight_layout(pad=2.0)

# Saving the current figure into the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Getting Series for alignment and interpolation inFigures 13, 14 and 15
wpi_yoy = df_wpi_ex_bonus_yoy_sa['Percentage Change From Corresponding Quarter of Previous Year']
unemployment_rate = df_labour_sa['Unemployment rate']
underemployment_rate = df_under_sa['Underemployment rate (proportion of labour force)']
underutilisation_rate = underemployment_rate + unemployment_rate
jv_rate = df_jv_rate_sa['Jobs']

# Convert unemployment_rate to quarterly (end of October)
unemployment_rate_q = unemployment_rate.resample('QS-OCT').mean()
underutilisation_rate_q = underutilisation_rate.resample('QS-OCT').mean()

# Figure 13: WPI vs. Unemployment - Since 2000
# Align the two series and filter the data
aligned_data_unemployment = pd.DataFrame(
    {'WPI YoY': wpi_yoy, 
    'Unemployment Rate': unemployment_rate_q}).dropna()
aligned_data_unemployment = aligned_data_unemployment[aligned_data_unemployment.index >= '2000-01-01']

# Calculate linear regression
slope, intercept = np.polyfit(aligned_data_unemployment['Unemployment Rate'], aligned_data_unemployment['WPI YoY'], 1)

# Create scatter plot with regression line
fig, ax = plt.subplots(figsize=(14, 8))

# Adding a page title above the plot
fig.suptitle('Figure 13: WPI vs. Unemployment - Since 2000 (Quarter)', fontsize=16, fontweight='bold', y=0.95)

# Scatter plot with regression line
sns.regplot(x='Unemployment Rate', y='WPI YoY', data=aligned_data_unemployment, scatter_kws={'color': 'tab:red'}, line_kws={'color': 'purple', 'linestyle': '--'}, ax=ax)

# Add title and labels
equation = f'y = {intercept:.4f} - {abs(slope):.4f}x'

# Add the line equation as the title of the graph
ax.set_title(equation, fontsize=12)
ax.set_xlabel('Unemployment Rate (%)')
ax.set_ylabel('WPI Ex−Bonus (Index) − YoY')

# Adding grid lines to the primary axis
ax.grid(True, alpha=0.4)

# Adjust layout to ensure the plot is centered and fits well within an A4 page
plt.tight_layout(pad=2.0)

# Save the current figure into the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 14: WPI vs. Underutilisation (UR + Underemployment) - Since 2000
# Align the series and filter the data
aligned_data_underutilisation = pd.DataFrame({'WPI YoY': wpi_yoy, 'Underutilisation Rate': underutilisation_rate_q}).dropna()
aligned_data_underutilisation = aligned_data_underutilisation[aligned_data_underutilisation.index >= '2000-01-01']

# Calculate linear regression
slope, intercept = np.polyfit(aligned_data_underutilisation['Underutilisation Rate'], aligned_data_underutilisation['WPI YoY'], 1)

# Create scatter plot with regression line
fig, ax1 = plt.subplots(figsize=(14, 8))

# Adding a page title above the plot
fig.suptitle('Figure 14: WPI vs. Underutilisation (UR + Underemployment) - Since 2000 (Quarter)', fontsize=16, fontweight='bold', y=0.95)

# Scatter plot with regression line
sns.regplot(x='Underutilisation Rate', y='WPI YoY', data=aligned_data_underutilisation, scatter_kws={'color': 'tab:red'}, line_kws={'color': 'purple', 'linestyle': '--'}, ax=ax1)

# Add title and labels
equation = f'y = {intercept:.4f} - {abs(slope):.4f}x'

# Add the line equation as the title of the graph
ax1.set_title(equation, fontsize=12)
ax1.set_xlabel('Underutilisation Rate (%)')
ax1.set_ylabel('WPI Ex−Bonus (Index) − YoY')

# Adding grid lines to the primary axis
ax1.grid(True, alpha=0.4)

# Adjust layout to ensure the plot is centered and fits well within an A4 page
plt.tight_layout(pad=2.0)

# Save the current figure into the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 15: WPI vs. Job Vacancy Rate - Since 2000
# Align the series and filter the data
aligned_data_job_vacancy = pd.DataFrame({'WPI YoY': wpi_yoy, 'Job Vacancy Rate': jv_rate}).dropna()
aligned_data_job_vacancy = aligned_data_job_vacancy[aligned_data_job_vacancy.index >= '2000-01-01']

# Calculate linear regression
slope, intercept = np.polyfit(aligned_data_job_vacancy['Job Vacancy Rate'], aligned_data_job_vacancy['WPI YoY'], 1)

# Create scatter plot with regression line
fig, ax2 = plt.subplots(figsize=(14, 8))

# Adding a page title above the plot
fig.suptitle('Figure 15: WPI vs. Job Vacancy Rate - Since 2000 (Quarter)', fontsize=16, fontweight='bold', y=0.95)

# Scatter plot with regression line
sns.regplot(x='Job Vacancy Rate', y='WPI YoY', data=aligned_data_job_vacancy, scatter_kws={'color': 'tab:red'}, line_kws={'color': 'purple', 'linestyle': '--'}, ax=ax2)

# Add title and labels
equation = f'y = {intercept:.4f} - {abs(slope):.4f}x'

# Add the line equation as the title of the graph
ax2.set_title(equation, fontsize=12)
ax2.set_xlabel('Job Vacancy Rate (%)')
ax2.set_ylabel('WPI Ex−Bonus (Index) − YoY')

# Adding grid lines to the primary axis
ax2.grid(True)

# Adjust layout to ensure the plot is centered and fits well within an A4 page
plt.tight_layout(pad=2.0)

# Save the current figure into the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Figure 16: Monthly Hours Worked
fig, ax1 = plt.subplots(figsize=(14, 8))

# Calculating series for plotting
hours_worked_yoy = percent_change(df_hour_worked_sa['Monthly hours worked in all jobs'])

# Add page title
fig.suptitle('Figure 16: Monthly Hours Worked', fontsize=16, fontweight='bold', y=0.95)

# Plot lines
ax1.bar(hours_worked_yoy.index, hours_worked_yoy, 25, label='YoY', color='tab:blue')

# Create a second y-axis for total unemployed
ax2 = ax1.twinx()
ax2.plot(df_hour_worked_sa['Monthly hours worked in all jobs'].index, df_hour_worked_sa['Monthly hours worked in all jobs'], label='Monthly Hours Worked (RHS)', color='tab:red', linestyle='-')

# Set labels for x and y axes
ax1.set_xlabel('Date')
ax1.set_ylabel('(%)')

ax2.set_ylabel('(Millions)')

# Set title for the axes
ax1.set_title('Monthly Hours Worked − YoY vs. Total')

# Format x-axis to show month/year
ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))

# Set x-axis limits
ax1.set_xlim([datetime(2009, 10, 1), hours_worked_yoy.index.max()])

# Add grid to the main axis
ax1.grid(True, alpha=0.4)

# Rotate x-axis labels
plt.setp(ax1.get_xticklabels(), rotation=90)

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Adjust layout to ensure the plot is centered and well-fitted on the A4 page
plt.tight_layout(pad=2.0)

# Save the current figure to the PDF
pdf.savefig(fig, bbox_inches='tight')

# Close the plot
plt.close()

# Closing the whole PDF
pdf.close()

print("Labour Market Report (ABS) generated!")

