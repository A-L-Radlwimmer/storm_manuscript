INCREASING WINTER STORMINESS IN THE SOUTHERN ROSS SEA, ANTARCTICA
Author: Antonia L. Radlwimmer
Date: 19.12.2025

This repository contains the python scripts used for the manuscript Radlwimmer et al., Increasing Winter Storminess in the Southern Ross Sea, Antarctica, submitted to Geophysical Research Letters in late 2025/early 2026. All scripts should be ready to run from the repository, it includes all necessary data and figures. It was written and created entirely by me, Antonia Radlwimmer. I used generative AI to help my coding. The folders contain the following:

Data
- si_file.csv ... comprehensive file containing temperature, pressure and storm index time series for all stations on a common time line
- atmospheric_modes ... folder containing SAM and SOI data from https://doi.org/10.6084/m9.figshare.24768654
- duffy ... folder containing data published by Duffy et al., 2024, at https://figshare.com/articles/dataset/_b_Duffy_et_al_b_-_Antarctic_polynya_database/24768654
- fraser ... folder containing formation timelines from Fraser.ipynb and csv-files imported from QGIS that were used in their creation
- ice ... folder containing ice fractions in McMurdo Sound
- folders named after weather stations
    - Scott Base files <station_number>__Pressure/Temperature__daily/hourly.csv were downloaded through NIWA data hub
    - files named <station_name>_3h.csv were downloaded from Wang et al., 2022, at https://doi.org/10.5194/essd-15-411-2023
    - files named <station_abbreviation><date>q1h.txt were downloaded from the AMRC archive using the scripts Scripts/Preprocess_Data/<station_name>.ipynb
    - any station's final data file preprocessed through Scripts/Preprocess_Data/<station_name>.ipynb is called <station_name>.csv

Figures
All figures produced by the scripts in this repository are saved in this folder. This folder also contains storm_paper_ice_ages(_2019-2024).png, storm_paper_ice_ages_legend.png, and storm_paper_map_complete.png which were produced in QGIS.

Scripts
All scripts can be run simply by restarting the kernel and running all cells in order (except for plot_2019_2024.ipynb, which needs the first part of the script to be run for each station before the plotting cells can be executed).
- Preprocess_Data ... folder containing preprocessing scripts for downloading data and bringing it into a uniform format
- Compute_and_Plot_SI.ipynb ... main storm index analysis; choose weather station in cell at beginning
- DataFile.ipynb ... generates Data/si_file.csv
- Duffy_and_SAM.ipynb ... analyzes relationship of Scott Base storm index and SAM and polynya activity
- duffy_wavelet_analysis_by_chatGPT.py ... contains functions used in wavelet analysis in duffy_and_SAM.ipynb
- Fraser.ipynb ... computes fast ice formation timelines from Fraser & Massom, 2020, fast-ice dataset (not included in repository, can be downloaded from Fraser, A.D. and Massom, R. (2020) Circum-Antarctic landfast sea ice extent, 2000-2018, Ver. 2.2, Australian Antarctic Data Centre - doi:10.26179/5d267d1ceb60c)
- plot_2019_2024.ipynb ... plots storm index time series for 2019-2024, illustration for seasonal storminess distribution and storm index-ice fraction relationship
- storm_indices.py ... contains functions used to compute the storm index and fit trend lines
- subplots.ipynb ... combines figures into figures with subplots