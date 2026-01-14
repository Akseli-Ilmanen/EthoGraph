## this script exemplifies how BirdPark data is loaded into python and prints file info
## author:  Linus Rüttimann
## version: 2023-01-01 
import h5py
import pandas as pd

# hdf5 file name
h5fn = 'Instances/copExpBP08/Data/2021-05-18/BP_2021-05-18_11-06-25_239375_0000000.h5'

# get file object
f1 = h5py.File(h5fn, 'r')

# read recording info
recInfo = f1['/recInfo'].attrs #is a dictionary of attributes

print('----recInfo:----')
for key in recInfo:
    print(key, ' : ', recInfo[key])

#read file info
fileInfo = f1['/fileInfo'].attrs #is a dictionary of attributes

print(' ')
print('----fileInfo:----')
for key in fileInfo:
    print(key, ' : ', fileInfo[key])

# show radio channel info
vNames = ("radioChNames", "radioChSenderColors", "radioChSenderIds", "radioChCenterFrqs_Hz", "radioChBirdRadioIds", "radioChSenderDescs")
radioChInfo = pd.DataFrame({k:recInfo[k] for k in vNames})

print(' ')
print('----radioCh info as a table:----')
print(radioChInfo)

# show daq channel info
daqChInfo = pd.DataFrame({k:recInfo[k] for k in ("daqChNames", "daqChDescs")})

print(' ')
print('----daqCh info as a table:----')
print(radioChInfo)
print(daqChInfo)

# example how to convert string timestamps to pandas datetime object
dateStrFormat = '%Y-%m-%dT%H:%M:%S.%f%z'
t = pd.to_datetime(recInfo['recStartTime'], format=dateStrFormat)

# read main data
radioSignals = f1['/radioSignals'][()] # accelerometer transmitter device signals (one row per channel)
daqSignals = f1['/daqSignals'][()] # microphone signals (one row per channel)

print(' ')
print('----main signals:----')
print(f'radioSignals:\tshape={radioSignals.shape}, dtype={radioSignals.dtype}')
print(f'daqSignals:\tshape={daqSignals.shape}, dtype={daqSignals.dtype}')

# auxFramesSignals
auxFrameSignals = f1['/auxFrameSignals']

print(' ')
print('----auxFrameSignals:----')
for key in auxFrameSignals:
    print(f'{key}:\tshape={auxFrameSignals[key].shape}, dtype={auxFrameSignals[key].dtype}')


# radioFrameSignals
radioFrameSignals = f1['/radioFrameSignals']

print(' ')
print('----radioFrameSignals:----')
for key in radioFrameSignals:
    print(f'{key}:\tshape={radioFrameSignals[key].shape}, dtype={radioFrameSignals[key].dtype}')
