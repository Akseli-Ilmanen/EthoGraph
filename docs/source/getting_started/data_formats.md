

### Other files needed

trialtree.md
labelling.md
changepoints.md
trials/metadata.md


## Data loading guide

`ethograph` allows you to visualize...
- videos from multiple camera angles (supported formats: `.mp4`, `.mov`, ...)
- overlaid pose estimation markers (DeepLabCut, Sleap, ... , {mod}`movement`)
- bioacoustics using an audio trace and spectrogram plot (LINK to audioio, e.g. `.wav`)
- raw single/ multi-channel electrophysiology data (formats supported by neo, `.dat`, ... - list phylib LINK)
- raster/PSTH plots of spike-sorted units (`kilosort` folder, `units` in `.nwb` files/{mod}`pynapple` files)
- custom **feature** data (pose estimation, kinematics, behavioural timeseries, neuronal firing rates, latent variables, model outputs, ...)

You can load your feature data either via {mod}`xarray`, {mod}`pynapple` or NWB (claude format this). Unlike, e.g. a `numpy` array, these data formats have explicit timestamps associated with each value. This allows `ethograph` to automatically align data formats of different sampling rates and modalities (video, audio & electrophysiology).

Besides visualization of feature data in a `LinePlot`, `HeatmapPlot`, and `SpacePlot`, `ethograph` has some functionalities that from a UI perspective are identical, but work differently under the hood depending on the data data format. 

1) **Feature dimension filtering**  

Example 1: You are working pose estimation data (e.g. DeepLabCut `.csv`/`.h5` files). You collected data from two individuals (`mouse_1`, and `mouse_2`), for 5 keypoints (`left_paw`, `right_paw`, `left_leg`, `right_leg`, `nose`) and across 3D space (`x`, `y`, `z`) and time. Besides the `position` feature, you also computed `velocity` (same dimensions), and `speed` (space dimension collapsed). 

In this example, the dimensions of our array are `(n_individuals, n_keypoints`, `space`, `time`). If we wanted to visualize a specific combination of these, we can use the mod movement and xarray sel valid (claude format this). 

claude add example for ds =load_from dlc path, sel across tehse dimesions, plot 

GIF - use DANNCE dataset with heatmaps

Claude mention how now using the combo selection/, all checkbox you can now visualize certain combinations (rephase this). See Advanced for how `ethograph` implements this for other data foramts



- Mention `sel_valid`, principlee.g. with speed not having space dims
- Label behaviours for `mouse_1` by selecting in dropdown

2) **Navigating by 'ethological trials'**

Example 1 continued: You are studying `mouse_1` and `mouse_2` because you are interested in the neural basis of social behaviours. Given your naturalistic set-up, you define trial periods by the following criterion: The two mice have been continuously less than 5cm apart for a minimum of 5seconds.  [why to define trials periods?] You have 7x 24h recordings, and social interactions are rare. Steps: 

1) Identify all start/stop periods of these social interactions
2) Save in a `.tsv` file with columns `start_time`, `stop_time`, and optional metadata, and load into GUI.
3) Use ``




4) **Navigating by behaviours/sequences**. 


Example 2: You are tool-use in crows.
Why 

Dandi, allows you to load a small subset of the data. 


3) **Trial filtering:**  


include GIF




### Data requirements 

ADD: image of get session, 



claude do tabs (similar to how it is in docs/installation.md), where for each tab we specify the requirements

xr.dataset 



### Advanced

Get sesion allowed objects


If you would look to understand 

|         | Description      | Relevant docs | xarray.Dataset | TrialTree | {mod}`pynapple` folder | `.nwb` file |
| ------- | ---------------- | ------------- | -------------- | --------- | ---------------------- | ----------- |
| Objects |                  |               |                |           |                        |             |
|         | Restrict in time |               |                |           |                        |             |
|         | sel valid        |               |                |           |                        |             |
|         |                  |               |                |           |                        |             |
|         |                  |               |                |           |                        |             |
|         |                  |               |                |           |                        |             |
|         |                  |               |                |           |                        |             |
|         |                  |               |                |           |                        |             |
