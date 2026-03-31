import ast
files = [
    "ethograph/gui/widgets_io.py",
    "ethograph/gui/widgets_data.py",
    "ethograph/gui/widgets_meta.py",
    "ethograph/gui/widgets_changepoints.py",
    "ethograph/gui/plots_container.py",
    "ethograph/gui/widgets_psth.py",
    "ethograph/gui/app_state.py",
    "ethograph/gui/wizard_single.py",
    "ethograph/gui/wizard_multi_tabs.py",
    "ethograph/gui/wizard_overview.py",
    "ethograph/gui/widgets_ephys.py",
]
for f in files:
    try:
        ast.parse(open(f).read())
        print(f"{f}: OK")
    except SyntaxError as e:
        print(f"{f}: SYNTAX ERROR: {e}")
