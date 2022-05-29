# rfi_filtering_2022
This repository contains main algorithms and machine learning model of Ventspils University of Applied Sciences Bachelor's degree thesis "Developing a methodology for processing radio astronomical data using machine learning algorithms for RFI
filtering". Other scripts for result analysis and other tests are not included as they cannot be used directly for different inputs.

**Included scripts and their parameter files:**
vdif_conversion.py - Converts data from VDIF format to Numpy NPY, splitting data as needed. File "vdif_params.json" is used as parameter input, example values are provided in this repo.

vdif_processing.py - Reads data from VDIF file, applies Fast Fourier Transform and displays results in all available channels. File "process_params.json" is used as parameter input, example values are provided in this repo.

denoise_prepare.py - Prepares data for created machine learning model training by creating necessary additional data. File "train_prepare_params.json" is used as parameter input, example values are provided in this repo.

rnn_full.py - Created machine learning model, based on Recurrent Neural Network. File "rnn_params.json" is used as parameter input, example values are provided in this repo.

denoise_generate.py - Generates data for model training by input parameters. File "generate_params.json" is used as parameter input, example values are provided in this repo.