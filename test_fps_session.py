import xarray as xr
from ethograph.io.trialtree import TrialTree

# Create a basic TrialTree
ds = xr.Dataset(attrs={'trial': 1, 'fps': 25})
dt = TrialTree()
dt['1'] = xr.DataTree(ds)

# Test backward compat: get_video_fps from ds.attrs
print('Backward compat fps:', dt.get_video_fps())

# Test set/get with cameras
dt.set_video_fps([30.0, 60.0], device_labels=['left', 'right'])
print('Left fps:', dt.get_video_fps('left'))
print('Right fps:', dt.get_video_fps('right'))
print('Default fps:', dt.get_video_fps())

# Test scalar set
dt2 = TrialTree()
dt2['1'] = xr.DataTree(xr.Dataset(attrs={'trial': 1}))
dt2.set_video_fps(25.0)
print('Scalar fps:', dt2.get_video_fps())
print('No camera fallback:', dt2.get_video_fps('nonexistent'))

# Test no fps at all
dt3 = TrialTree()
dt3['1'] = xr.DataTree(xr.Dataset(attrs={'trial': 1}))
print('None fps:', dt3.get_video_fps())

# Test set with device_labels but scalar fps
dt4 = TrialTree()
dt4['1'] = xr.DataTree(xr.Dataset(attrs={'trial': 1}))
dt4.set_video_fps(30.0, device_labels=['cam1', 'cam2'])
print('cam1 fps:', dt4.get_video_fps('cam1'))
print('cam2 fps:', dt4.get_video_fps('cam2'))

print('\nAll tests passed!')
