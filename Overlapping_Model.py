# ॐ नमः सिद्धम् 
# "Unsupervised ML model to find patterns like overlapping timings with how many people working in those timings,
# start & end of overlapping time, duration in hrs & show the best time slot of 1 hour when entire different teams are active
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans

# loading the csv File 
data = pd.read_csv('D:\\wlc\\task2.csv')  

# Convert the 'start time' and 'end time' columns to datetime object
data['start time'] = pd.to_datetime(data['Start Time'], format='%H:%M')
data['end time'] = pd.to_datetime(data['End Time'], format='%H:%M')

# Function to find overlapping time and calculate duration
def find_overlaps(data):
    intervals = data[['start time', 'end time']].values

    # k-means clustering with k=2 
    kmeans = KMeans(n_clusters=3)
    data['cluster'] = kmeans.fit_predict(intervals)

    # overlapping intervals
    overlapping_intervals = data[data['cluster'] == 0]
    overlapping_intervals = overlapping_intervals.sort_values(by='start time')

    # duration of overlap
    overlapping_intervals['overlap_duration'] = (overlapping_intervals['end time'].shift(-1) - overlapping_intervals['start time']).fillna(pd.Timedelta(seconds=0))


    return overlapping_intervals

overlapping_times = find_overlaps(data)
print(overlapping_times)
