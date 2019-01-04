#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:40:59 2018

@author: ftang1
"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

pd.set_option('display.max_columns', 15)
plt.rcParams["patch.force_edgecolor"] = True

# =============================================================================
# Read in raw data
# =============================================================================

xls = pd.ExcelFile("data.xlsx")
trips = xls.parse()

print('trips cols: ', list(trips.columns.values))
print('shape: ', trips.shape)

# =============================================================================
# Clean & pre-process data
# =============================================================================

# Calculate trip duration
trips['trip_duration'] = trips['end_time'] - trips['start_time']

print("Looking at trip_duration, I noticed that certain Timedelta values are \
negative. This shouldnt happen because end_time should always take place \
after start_time.")

# Print only rows with "negative" time_duration values to compare end_time and start_time.
neg_delta = trips['trip_duration'] < pd.Timedelta(0)
print("Trips with negative delta: \n", trips[neg_delta])
print("Number of trips with negative delta: ", len(trips[neg_delta])) # returns 15
print("If trip starts before midnight (start_time) and ends after midnight on \
subsequent day (end_time), then the date object in end_time does not update \
properly.")

# Fix end_time values
def fix_negs(df):
    '''Expects a dataframe with columns 'end_time' and 'start_time.'
    Searches for trips with end_time occurring before start_time. If erroneous 
    end_time is found, it adds 1 day. Returns the correct end_time.'''
    if df['end_time'] < df['start_time']:
        return df['end_time'] + dt.timedelta(days=1)
    else:
        return df['end_time']

# Apply fix_negs() on dataframe by row-level
trips['correct_end_time'] = trips.apply(fix_negs, axis=1)
print("\nIncorrect end_time values are fixed.")
 
# Fix trip_duration values by recalculating with corrected end_time values
trips['trip_duration'] = trips['correct_end_time'] - trips['start_time']

# Verify negative timedeltas have been fixed
neg_delta = trips['trip_duration'] < pd.Timedelta(0)
print("\nTrips with negative delta \n", trips[neg_delta]) # returns empty df
print("Number of trips with negative delta: ", len(trips[neg_delta])) # returns 0

# Drop original end_time column
trips.drop(['end_time'], axis=1, inplace=True)
trips.rename(columns={'correct_end_time':'end_time'}, inplace=True)
print('shape: ', trips.shape)
print('trips cols: ', list(trips.columns.values))

# =============================================================================
# Number of near misses & car_id
# =============================================================================

print("\nAre certain car types (car_id) safer than others (num_near_misses)?")
num_misses_per_car = trips.groupby(['car_id']).sum()['num_near_misses']
print('\nNumber of near misses per car_id \n', num_misses_per_car)
print('Superman has 23 near misses, while the other car_id types have 0, 1, or 2 \n')

print("\nBut perhaps superman had more trips than the other car types, in which case, superman would not necessarily be more dangerous than other car types.")

print("Near-miss rates per car_id would be more helpful to look at.")
# Calculate total trips per car_id
# Here, I am assuming there is only 1 physical car per car_id
num_trips = trips.groupby(['car_id']).count()['start_time']
print('\nNumber of trips per car_id \n', num_trips)

# Calculate "near-accident" rates
print('Near-miss rates per car_id: ', num_misses_per_car.values/num_trips.values)
print("These percentages indicate that superman's higher num of near misses is not due to its greater total number of trips taken.")


print("\nSo is superman more dangerous (i.e. its features or automotion) than the others?")
print("Let's explore some confounding variables: hour of day & trip duration, both of which could influence superman's number of near misses.")

print("\nPerhaps superman's high number of near misses is due to having taken trips during traffic hours when accidents are more likely to occur, such as peak traffic times.")

# Line Plot: almost-accidents (num_near_misses) over time (start_time)
# Since start_time is always on 10/2/2018, we only need the time portion
trips['start_time_of_trip'] = pd.to_datetime(trips['start_time'], unit='s').dt.time
superman = trips.where(trips['car_id'] == 'superman')
superman.dropna(axis=0, how='all', inplace=True)
table = pd.pivot_table(superman, values='num_near_misses', index='start_time_of_trip', columns='car_id')
table.plot(figsize=(10,5), title="Superman's Near Misses by Hour of Day", legend=False, xticks=('03:00','05:30','08:00','10:30','13:00','15:30','18:00','20:30','23:00'))

print("Looking at superman's near misses in this plot, this seems unlikely -there are near-misses occurring throughout the day for superman.")

# Overlaid histograms on trip duration per car_id
trips['tripdur_sec'] = trips['trip_duration'].astype('timedelta64[s]')

df0 = trips[trips['car_id'] == 'spiderman']['tripdur_sec']
df1 = trips[trips['car_id'] == 'superman']['tripdur_sec']
df2 = trips[trips['car_id'] == 'hulk']['tripdur_sec']
df3 = trips[trips['car_id'] == 'scarecrow']['tripdur_sec']
df4 = trips[trips['car_id'] == 'venom']['tripdur_sec']
df5 = trips[trips['car_id'] == 'robin']['tripdur_sec']
df6 = trips[trips['car_id'] == 'batman']['tripdur_sec']
df7 = trips[trips['car_id'] == 'joker']['tripdur_sec']
df8 = trips[trips['car_id'] == 'ironman']['tripdur_sec']

fig, ax = plt.subplots(figsize=(10,5))
ax.hist(df0, alpha=0.5, histtype='step', label='spiderman')
ax.hist(df1, alpha=0.5, label='superman')
ax.hist(df2, alpha=0.5, histtype='step', label='hulk')
ax.hist(df3, alpha=0.5, histtype='step', label='scarecrow')
ax.hist(df4, alpha=0.5, histtype='step', label='venom')
ax.hist(df5, alpha=0.5, histtype='step', label='robin')
ax.hist(df6, alpha=0.5, histtype='step', label='venom')
ax.hist(df7, alpha=0.5, histtype='step', label='joker')
ax.hist(df8, alpha=0.5, histtype='step', label='ironman')

ax.legend(loc='upper right')
plt.title('Trip Duration by car_id')
plt.ylabel('Number of Trips')
plt.xlabel('Trip Duration (seconds)')
plt.savefig('tripdur')
plt.show()
plt.close()

print("\nIs there a relationship between trip duration and the number of times that car is close to getting into an accident?")
print("Perhaps number of near misses is positively correlated with length of trip, and superman happened to take longer trips than the other cars.")

print("Given the overlaid histograms, this doesn't seem to be the case - superman's trips are not noticeably longer than that of other cars.")

print('\nMedian trip duration (seconds) per car_id \n', trips.groupby('car_id').median()['tripdur_sec'])
print("Batman, robin, and spiderman have longer median trip time compared to superman.")

print("\nWe can infer that superman is more dangerous given that we cannot attribute its higher number of near misses to external factors, such as hour of day and trip duration.")
print("Recommendation to Ryde Autoamtion: look into specific features of superman vehicle, i.e. perhaps its sensors are broken.")

# =============================================================================
# Trip Duration: Percentiles
# =============================================================================

print("\nWhat is the mean and median trip duration?")
print('The average trip duration is ', trips['trip_duration'].mean())
print('The median trip duration is ', trips['trip_duration'].median())

# Trip duration for entire dataset (all cars)
fig, ax1 = plt.subplots(figsize=(10,5))
s=trips['tripdur_sec']
values, bins, _ = plt.hist(s, density=True)
plt.title('Trip Duration (all cars)')
plt.ylabel('Frequency')
plt.xlabel('Trip Duration (seconds)')
plt.savefig('tripdur_all')
plt.show()
plt.close()

def calculate_value_at_percentile(series, pct):
    v = series.quantile(pct)
    val_min = int(v // 60)
    val_sec = int(v % 60)
    print(pct*100, '% of the trips had a trip duration equal to or less than', val_min, 'minutes and', val_sec, 'seconds.')
    
calculate_value_at_percentile(s, .25)
calculate_value_at_percentile(s, .5)
calculate_value_at_percentile(s, .75)


# =============================================================================
# Price
# =============================================================================

# Prices over the course of the day - by car_id
trips['start_hour_of_trip'] = pd.to_datetime(trips['start_time'], unit='s').dt.hour
table1 = pd.pivot_table(trips, values='price', index='start_hour_of_trip', columns='car_id')
table1.plot(figsize=(10,5), title="Price of Trips Over the Course of the Day")
plt.savefig('price_over_day_carid')

# Prices over the course of the day - all trips
trips['start_hour_of_trip'] = pd.to_datetime(trips['start_time'], unit='s').dt.hour
table2 = pd.pivot_table(trips, values='price', index='start_hour_of_trip')
table2.plot(figsize=(10,5), title="Price of Trips Over the Course of the Day")
plt.savefig('price_over_day')

print("\n The price dropped shortly after 7:30 AM and peaked at around 2:00 PM.")

# Price Range grouped by number of riders
fig, ax2 = plt.subplots(1,1,figsize=(10,5))
sns.set_style("whitegrid")
sns.boxplot(x='num_riders', y='price', data=trips, ax=ax2).set_title('Price Range by num_riders')
medians = trips.groupby(['num_riders'])['price'].median().values # calculate medians
median_labels = [str(np.round(s, 2)) for s in medians] # add median labels
pos = range(len(medians))
for tick, label in zip(pos, ax2.get_xticklabels()):
    ax2.text(pos[tick], medians[tick] + 0.05, median_labels[tick], horizontalalignment='center', size='large', color='w', weight='semibold')
plt.yticks([2,3,4,5,6],['$2','$3','$4','$5','$6'])
plt.savefig('bw_price_bynumriders')
plt.show()
plt.close()

print("The price range for trips with only 1 rider is significantly higher than that of trips with multiple riders.")
print("That said, the profitability of trips with 1 rider is mitigated by the fact that 'price' is a per-rider price.")

# Calculate total revenue grouped by num_rider 
trips['rev_trip'] = trips['price']*trips['num_riders'] # create new col with revenue earned for that single trip

def calculate_revenue(df, n):
    riders = df[df['num_riders'] == n] # filter for rows with n num_riders
    return riders['rev_trip'].sum()

rev1 = calculate_revenue(trips, 1) # 1114.34
rev2 = calculate_revenue(trips, 2) # 1047.66
rev3 = calculate_revenue(trips, 3) # 1692.57
rev4 = calculate_revenue(trips, 4) # 2119.04
rev5 = calculate_revenue(trips, 5) # 3512.95

print('\nTotal revenue generated from trips with 1 rider: $', np.round(rev1))
print('Total revenue generated from trips with 2 riders: $', np.round(rev2))
print('Total revenue generated from trips with 3 riders: $', np.round(rev3))
print('Total revenue generated from trips with 4 riders: $', np.round(rev4))
print('Total revenue generated from trips with 5 riders: $', np.round(rev5))

# Since each record in dataset is a trip, we can simply count the rows to get total trips for each num_riders group
t = trips.groupby(['num_riders']).size()
num_trips1r, num_trips2r, num_trips3r, num_trips4r, num_trips5r = t.values[0], t.values[1], t.values[2], t.values[3], t.values[4]
print('\nTotal trips with 1 rider: ', num_trips1r) # 196
print('Total trips with 2 riders: ', num_trips2r) # 194
print('Total trips with 3 riders: ', num_trips3r) # 203
print('Total trips with 4 riders: ', num_trips4r) # 190
print('Total trips with 5 riders: ', num_trips5r) # 250
print("Approx 200 total trips for each group of num_riders (except trips with 5 riders, which has 250 trips), so it's about equal across the board.")
      
# Average revenue generated per trip
print('\nAverage revenue generated per trip:')
print('Trips with 1 rider: $', round(rev1/num_trips1r, 2)) # 5.69
print('Trips with 2 riders: $', format(round(rev2/num_trips2r, 2), '.2f')) # 5.40
print('Trips with 3 riders: $', round(rev3/num_trips3r, 2)) # 8.34
print('Trips with 4 riders: $', round(rev4/num_trips4r, 2)) # 11.15
print('Trips with 5 riders: $', round(rev5/num_trips5r, 2)) # 14.05
 

print("\nEven though the price is significantly higher for trips with 1 rider, trips with greater num of riders are more profitable.")
print("This is because num_riders is a multiplier when calculating revenue generated.")

print("\nThe total revenue for trips with 2 riders was less than that of trips with 1 rider.")
print("This makes sense because the price/rider is multiplied only by 2 (rather than by 3,4,5). Because multiplier is 2, it makes sense that the total revenue breaks even with trips with 1 rider, which is set at double the price but has half the number of riders per trip.")

print("\nThe data suggests that trips with 1 rider vs. 2 riders are approximately equally profitable.")
print("Trips with 3,4,5 riders are more profitable than trips with 1 or 2 riders.")
print("Recommendation to Ryde Autoamtion: Enhance marketing/sales to increase trips taken in groups of 3, 4, or 5 riders.")

# Box and whisker plot: Price Range grouped by car_id
fig, ax3 = plt.subplots(1,1,figsize=(10,5))
sns.set_style("whitegrid")
sns.boxplot(x='car_id', y='price', data=trips, ax=ax3).set_title('Price Range by car_id')
medians = trips.groupby(['car_id'])['price'].median().values # calculate medians
median_labels = [str(np.round(s, 2)) for s in medians] # add median labels
pos = range(len(medians))
for tick, label in zip(pos, ax3.get_xticklabels()):
    ax3.text(pos[tick], medians[tick] + 0.05, median_labels[tick], horizontalalignment='center', size='large', color='w', weight='semibold')
plt.yticks([2,3,4,5,6],['$2','$3','$4','$5','$6'])
plt.savefig('bw_price_bycar')
plt.show()
plt.close()

print("All cars have similar medians. All cars have similar distribution, except for joker. Joker's third and fourth quartiles are higher than the rest. ")

# Price histogram - all records
fig, ax4 = plt.subplots()
prices=trips['price']
values, bins, _ = plt.hist(prices, density=True)
plt.title('Price - all trips')
plt.ylabel('Frequency')
plt.xticks([2,3,4,5,6],['$2','$3','$4','$5','$6'])
plt.xlabel('Price')
plt.savefig('hist_prices')
plt.show()
plt.close()

print("\nThe histogram displays a bimodal distribution of prices, as we could have inferred from the 'Price Range per num_riders' box plot.")

upper = format(trips[trips['num_riders'] == 1]['price'].min(), '.2f') # $5.00
lower = format(trips[trips['num_riders'] != 1]['price'].max(), '.2f') # $3.50
print(f"There are no prices between ${lower} and ${upper} in entire dataset.")
 
print(f"\nRecommendation to Ryde Autoamtion: Test/experiment with the price range between ${lower} and ${upper}. Behavioral economics-- if we raise prices by $1-2 for trips with 2, 3, 4, and 5 riders, would we see a decrease in trips taken?")

# =============================================================================
# Rating
# =============================================================================

# Pie Chart on Ratings
labels = '3', '5', '4', '2', '1'
sizes = trips['rating'].value_counts()
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'cyan']
explode = (0.1, 0, 0, 0, 0)  # explode 1st slice
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Rider Ratings (5 is best)')
plt.axis('equal')
plt.savefig('pie_ratings')
plt.show()
plt.close()

# Rating histogram - all records
fig, ax8 = plt.subplots()
prices=trips['rating']
values, bins, _ = plt.hist(prices, density=True, color='g', bins=5)
plt.title('Rating - all trips')
plt.ylabel('Frequency')
plt.xticks([1, 2,3,4,5,],['1','2','3','4','5'])
plt.xlabel('Rating')
plt.savefig('hist_ratings')
plt.show()
plt.close()

# Regression Plot: relationship between number of near-accidents and rating
ax5 = sns.regplot(x=trips['num_near_misses'], y=trips['rating'], data=trips, color="g", fit_reg = True)
plt.title('Number of Near-Misses and Rating')
plt.savefig('regplot_acc-rating')
plt.show()
plt.close()

# Regression Plot: relationship between price and rating
ax6 = sns.regplot(x=trips['price'], y=trips['rating'], data=trips, color="g", fit_reg = True)
plt.title('Price and Rating')
plt.savefig('regplot_price-rating')
plt.show()
plt.close()

# Price Range grouped by ratings
fig, ax7 = plt.subplots(1,1,figsize=(10,5))
sns.set_style("whitegrid")
sns.boxplot(x='rating', y='price', data=trips, ax=ax7).set_title('Price Range and Rating')
medians = trips.groupby(['rating'])['price'].median().values # calculate medians
median_labels = [str(np.round(s, 2)) for s in medians] # add median labels
pos = range(len(medians))
for tick, label in zip(pos, ax7.get_xticklabels()):
    ax7.text(pos[tick], medians[tick] + 0.05, median_labels[tick], horizontalalignment='center', size='large', color='w', weight='semibold')
plt.yticks([2,3,4,5,6],['$2','$3','$4','$5','$6'])
plt.savefig('price_byrating')
plt.show()
plt.close()


print("Rating of 3 has highest percentage, at 41.7%. 91.28% of trips were given a rating of 3 or higher.")
print("There is a clear negative correlation between number of near misses and rating.")
print("There seems to be no significant correlation between price and rating.")

# Ratings over the course of the day - by car_id
table3 = pd.pivot_table(trips, values='rating', index='start_hour_of_trip', columns='car_id')
table3.plot(figsize=(10,5), title="Ratings Over the Course of the Day")
plt.savefig('ratings_over_day_bycar')

# Ratings over the course of the day - all trips
table4 = pd.pivot_table(trips, values='rating', index='start_hour_of_trip')
table4.plot(figsize=(10,5), title="Ratings Over the Course of the Day")
plt.savefig('ratings_over_day')

print("\nThere was a drop in ratings at around 3:00 PM.")