# Artifact Removal Tool Validation

## Summary
This repo contains the validation study for the eye-blink artifact removal tool. The repo is organized in the following order: 
- [Data](./Data/): Store the data used to validate the tool here. In the examples provided, we are using the Temple Artifact Data for validation.
- Functions
    - [Artifact removal tools](). This tool was originally designed to eliminate eye-blinks from single channel EEG.
    - [EEG quality index](). This tool was designed to quantify how clean a certain dataset is. The tool requires to provide a "clean" dataset for comparison purposes.
- [Notebooks](). This folder contains the Jupyter Notebooks to run the study. Each notebook is labeled with a number in its name to show the order in which they should be run

## Dependencies
Make sure to install the [bci_art.yml](./bci_art.yml) virtual environment. This can be installed from the terminal using:

> conda env create -f bci_art.yml

## Methodology


## Results

## Notes
- The artifact removal tool and EEG quality index were taken from their respective repositories, make sure that the version of those functions matches the version in this repository.
