"""Minimal repro: build a window, load birdpark, close it, N times.

  python repro.py <tmpdir> <n> [novideo]
"""
import sys
from pathlib import Path


def main():
    from qtpy.QtWidgets import QApplication
    import ethograph.utils.paths as paths_module
    from ethograph.gui.main_window import EthographMainWindow
    from ethograph.gui.widgets_meta import MetaWidget
    from ethograph.datasets import resolve_dataset_paths

    novideo = "novideo" in sys.argv
    app = QApplication.instance() or QApplication([])
    tmp = Path(sys.argv[1]); tmp.mkdir(parents=True, exist_ok=True)
    cfg = tmp / ".ethograph"; cfg.mkdir(exist_ok=True)
    paths_module.default_config_dir = lambda d=None: cfg
    r = resolve_dataset_paths("birdpark")
    keys = ("audio_folder", "pose_folder") if novideo else ("video_folder", "audio_folder", "pose_folder")

    for i in range(int(sys.argv[2])):
        shell = EthographMainWindow(); meta = MetaWidget(shell); shell.attach_meta_widget(meta)
        meta._check_unsaved_changes = lambda e: True
        meta.app_state._layout_snapshot_provider = None
        io = meta.io_widget; io._clear_all_line_edits()
        io.nc_file_path_edit.setText(r["nc_file_path"]); meta.app_state.nc_file_path = r["nc_file_path"]
        for k in keys:
            if r.get(k): getattr(io, f"{k}_edit").setText(r[k]); setattr(meta.app_state, k, r[k])
        meta.data_widget.on_load_clicked(); QApplication.processEvents()
        shell.close(); QApplication.processEvents()
        print(f"  cycle {i + 1} ok", flush=True)
    print("SURVIVED novideo=" + str(novideo))


if __name__ == "__main__":
    main()
