import pandas as pd
import argparse

from util import *


# parse filename
parser = argparse.ArgumentParser(description='Kalman Filter for RSSI Time Series')
#parser.add_argument('--file', nargs='?', help='data filename', default='../data/sample.csv')
parser.add_argument('--file', nargs='?', help='data filename', default='../data/Long_Traj_344_Filter_1ms_RSSI.csv')

args = parser.parse_args()

file_name = args.file

# open file and read RSSI signals (average of the 11 RSSI Signals)
file = pd.read_csv(file_name)
signal = file['rssi']

# calculate filters
signal_kalman_filter = kalman_filter(signal, A=1, H=1, Q=1.6, R=6)


# plot signal and filters
plot_signals([signal, signal_kalman_filter], ['signal','kalman_filtered_signal'])