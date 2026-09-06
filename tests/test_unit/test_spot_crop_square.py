"""The crop tool's *Square* option: what is drawn is what the extractor sees.

The video-feature extractors centre-crop a square, so the tool grows the
dragged rectangle to a square about its centre. Qt-free: the arithmetic is
one function.
"""

from ethograph.gui.dialog_spot_crop import square_box


def test_grows_to_the_longer_side_about_the_centre():
    # 203x164 at (164, 0): side 203, centre (265.5, 82) → y would start at -19.5, shifted to 0
    assert square_box(164, 0, 367, 164, 640, 480) == (164, 0, 367, 203)


def test_a_square_is_unchanged():
    assert square_box(10, 20, 110, 120, 640, 480) == (10, 20, 110, 120)


def test_shifted_back_inside_the_frame():
    # a wide box against the right edge: the square must not overhang the frame
    assert square_box(500, 200, 640, 260, 640, 480) == (500, 160, 640, 300)


def test_shrunk_only_when_the_frame_is_too_small():
    # a 300x50 box in a 300x100 frame: the square can only be 100, centred on the box
    assert square_box(0, 0, 300, 50, 300, 100) == (100, 0, 200, 100)


def test_the_drag_itself_is_held_square():
    """With Square ticked the cursor's far corner moves the same amount on both axes, direction kept."""
    from ethograph.gui.pygfx_video import square_corner

    assert square_corner(100, 100, 180, 130) == (180, 180)  # right-down, wider than tall → square on width
    assert square_corner(100, 100, 120, 170) == (170, 170)  # taller than wide → square on height
    assert square_corner(100, 100, 40, 130) == (40, 160)  # dragging down-left stays down-left
    assert square_corner(100, 100, 40, 70) == (40, 40)  # dragging up-left stays up-left
    assert square_corner(100, 100, 150, 50) == (150, 50)  # already square: unchanged
