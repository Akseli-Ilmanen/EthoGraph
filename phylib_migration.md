claude pls do a migration 

currently in @plots_ephystrace.py, we mainly have a generic ephys loader using Neo. 
I would like to maintin this loader, but also add a phylib_loader taht is very simple, only with info:
n_channels_dat = 64
offset = 0
sample_rate = 30000
dtype = 'int16'
dat_path = 'C:\\Users\\Admin\\Documents\\Akseli\\AI_data\\rawdata\\sub-02_id-Poppy\\ses-000_date-20260309_01\\ephys\\Poppy_260309_111034\\amplifier.dat'

from phylib.io.traces import get_ephys_reader
reader = get_ephys_reader(path, sample_rate=30000, n_channels=64)
_ = reader[:1000]






Neo-Viewer
- add a combo to data widget, where user can select teh different streams from generic ephys loader, 
- shows streams as pyqt graph, if there is a stream that exactly matches the n_channels ´m sample rate of params.poy (below), show thsi stream in combo but keep it greyed out, other streams will be things liek auxillary but you can just use neo to figure out stream names (code should alreayd exist)

IO widget file path: "Ephys_file"


Phy-Viewer
- everythiing currently in widget_ephys about selecting channels, showing good neurons, etc


if kilosort folder provied, look for params.py file, with dat_path = to populate dat_path for 
If no params.py file exists an, create a pop up prompting the user to fill in the follwiing info
n_channels_dat = None (default)
offset = 0 (default)
sample_rate = 30000 (default)
dtype = 'int16' 
dat_path = 'C:\\Users\\Admin\\Documents\\Akseli\\AI_data\\rawdata\\sub-02_id-Poppy\\ses-000_date-20260309_01\\ephys\\Poppy_260309_111034\\amplifier.dat'
(path empty default)

alterantively if dat_path does not exist on machien, prompt user to update params.py, 
- when clicked, save the params.py file in kilosort folder 


when you ahve this done, you also need to update @plots_ephystrace.py, we can simplify a lot
e.g. we can use below to index an interval in seocnds

def select_traces(traces, interval, sample_rate=None):
    """Load traces in an interval (in seconds)."""
    start, end = interval
    i, j = round(sample_rate * start), round(sample_rate * end)
    i, j = int(i), int(j)
    traces = traces[i:j]
    traces = traces - np.median(traces, axis=0)
    return traces


in many cases, you can simplify things, for example remove the sharedephys cache everweyhwere in @gui, and simplify or even get rid of the complex buffering logic, instead try to more mimic what phy is doing ref phy

Note that sionce we have 2 panels, in data widget we need functionaltiy (2 chekcboxes) for which is shown, aloso make data loading code is flexible if the user does not specify not specify ephys file (for neo), or alternatively does not specify kilosort folder, everything should still work,and the entier code in @widgets_ephys is related to the phy-viewer/panel, the Neo viewer is very simple, no added control,

GENERALLY, dont ask fpr mpersmission just change, and try to simplify the code a lot using more phy native methods, direct indexing, also ref phylib phylib.io.traces, are some functions like get_spike_waveforms whichn you can use, try to copy code as much from phy as possible dont reinvent the wheel 