:hide-toc:

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   advanced/index
   examples/index
   api_index
   community/index

ethograph
=========

EthoGraph is a graphical user interface for visualizing and segmenting
multimodal timeseries behavioural data. It builds upon a number of :ref:`open-source
libraries <target-support>` to load and quickly render video and pose files, audio and
spectrograms, ephys recordings in various formats, and arbitrary
multi-dimensional timeseries — all on one shared timeline.

Note this GUI is still in development. I welcome people testing it and
:doc:`providing feedback <community/contributing>`!

.. raw:: html

   <style>
     .vimeo-wrapper { width: 100%; }
     @media (min-width: 768px) {
       .vimeo-wrapper { width: 65%; margin: 0 auto; }
     }
   </style>
   <div class="vimeo-wrapper">
     <div style="padding:56.25% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/1206424641?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media; web-share" referrerpolicy="strict-origin-when-cross-origin" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="Ethograph Demo"></iframe></div>
   </div>
   <script src="https://player.vimeo.com/api/player.js"></script>

Quickstart
----------


To install the GUI, run the following. For more detailed instructions, see
:doc:`installation <getting_started/installation>`

.. code-block:: bash

   uv pip install "ethograph[gui,audio]" --python 3.12

To open the GUI, run:

.. code-block:: bash

   ethograph launch

After launching, there are some :doc:`example datasets <examples/index>` you can explore the GUI. To
learn more about all the functionalities, I recommend the
:doc:`user manual <getting_started/user_manual>`.

.. _target-support:

Support
-------

.. image:: media/opensource.png
   :alt: Open-source projects EthoGraph depends on
   :align: left
   :width: 60%

EthoGraph is built on top of a number of open-source projects:
`PyAV <https://pyav.org/docs/stable/>`_,
`audioio <https://github.com/bendalab/audioio>`_,
`Neo <https://neo.readthedocs.io>`_,
`crowsetta <https://github.com/vocalpy/crowsetta>`_,
`Neurodata Without Borders <https://www.nwb.org/>`_,
`xarray <https://docs.xarray.dev/>`_,
`pynapple <https://pynapple.org/index.html>`_,
`movement <https://movement.neuroinformatics.dev/>`_,
`phy <https://github.com/cortex-lab/phy>`_,
`PyQtGraph <https://www.pyqtgraph.org/>`_, and
`pygfx <https://pygfx.org/>`_ (via
`pynaviz <https://github.com/pynapple-org/pynaviz>`_).
