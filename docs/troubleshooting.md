# Troubleshooting

## Error Messages

1. If you run into a problem, I recommend `Ctrl + S` to save labels `.nc` file, and then restart the GUI. Generally recommended to save semi-regularly!
2. If you are starting a new project with different type of data, it may be helpful to click on the `Reset gui_settings.yaml` to avoid `KeyError`/`AttributeError` from previously selected `..._sel` config.


## FAQ

### Opening .tsv label files in Excel

Excel on Windows may not correctly parse `.tsv` files when double-clicked due to regional delimiter settings. To open correctly:

1. Open Excel
2. **File → Open → Browse**
3. Change file filter to **"All Files (\*.\*)"**
4. Select the `.tsv` file
5. Excel will launch the Text Import Wizard — select **"Tab"** as delimiter


## Issues

Feel free to raise any GitHub issues [here](https://github.com/Akseli-Ilmanen/EthoGraph/issues).