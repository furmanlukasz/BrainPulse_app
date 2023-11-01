# BrainPulse Playground

## Overview
This is a Python application using Streamlit, designed to showcase the analysis of EEG signals. It performs several complex computations like STFT (Short Time Fourier Transform), RQA (Recurrence Quantification Analysis), and various other matrix operations on the EEG data.

## Features
* User interface via Streamlit for easier interaction
* Customizable options for electrode selection, time range, and FFT parameters
* Data visualization using Matplotlib and Plotly
* Save & Download feature for analyzed data
* Modular codebase for easier understanding and extensibility

## Dependencies
* Python 3.x
* Streamlit
* Matplotlib
* Plotly
* NumPy
* Pandas

## Installation
To install the dependencies, run:
```bash
pip install streamlit matplotlib plotly numpy pandas
```

## How to Use
1. Clone the repository to your local machine.
2. Navigate to the directory and run `streamlit run <filename.py>`
3. Use the Streamlit sidebar to choose your settings and analyze the EEG data.

## Functionality Breakdown
1. **dataset.eegbci_data**: Fetch EEG BCI data for selected time and subject.
2. **vector_space.compute_stft**: Compute STFT for given settings.
3. **distance_matrix.EuclideanPyRQA_RP_stft_cpu**: Calculate distance matrix using Euclidean distance.
4. **recurrence_quantification_analysis.get_results**: Perform RQA analysis on the calculated matrix.
5. **plot_rqa**: Plot RQA data on radar chart.
6. **waterfall_spectrum**: Plot waterfall spectrum of STFT.
7. **save**: Save the analyzed matrices as `.npy` files.
8. **download**: Download the saved matrices as a `.zip` file.

## License
This project is open-source and available under the [MIT License](LICENSE).
