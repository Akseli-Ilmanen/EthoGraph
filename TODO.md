

### Tests
- add dt = eto.from_datasets(ds_list) for testing (things shouldnt break here!)

-     dt = eto.dataset_to_basic_trialtree(
        ds, video_path=video_path, video_motion=video_motion
    )
    - replace all these instances with jsut a xr.ds, and use continuous slicing if user wants to add trials table here.