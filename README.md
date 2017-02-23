Smoothed Dyadic Partitions
----------------------------------------------
Deep nonparametric conditional discrete probability estimation via smoothed dyadic partitioning.

- See the supplementary file (paper/supplementary.pdf) for the additional plots.

- See the `tfsdp` directory (specifically models.py) for the details of all the models we implemented. Note that this file contains a lot of garbage code and legacy naming that needs to be cleaned up. Our SDP model is named LocallySmoothedMultiscaleLayer and is often referred to in some of the experiments as trendfiltering-multiscale. 

- See the `experiments` directory for the code to replicate our experiments.

Note that you should be able to install the package as a local pip package via `pip -e .` in this directory. The best example for how to run the models is in `experiments/uci/main.py`, which contains the most recent code and should not have any API issues.

Citation
========
If you use this code in your work, please cite the following:
