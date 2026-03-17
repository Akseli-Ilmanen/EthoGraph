Add some notes about NWB integration.

What I noticed: 
- Downloading individual trials of pose data (for pose overlay) or general feature data for a trial slice is quite slow. And downloading this data from dandi should be relatively fast (max few mins). Similarly, behavioural intervals (labels), should be saved 


- But downloading ephys, video or audio would take much longer 20min/hours. The user should be able to download these, if he wants to (and e.g. stream ephys from local .nwb file), but this optional. 