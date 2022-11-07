from cProfile import run
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from complexRadar import ComplexRadar
import math
from zipfile import ZipFile
from glob import glob
import os
from BrainPulse import (dataset,
                        vector_space,
                        distance_matrix,
                        recurrence_quantification_analysis,
                        features_space,
                        plot)

# path
path = "./mne_data"
path2 = "./RPs"

# Remove the specified
# file path
try:
    os.remove(path)
    print("% s removed successfully" % path)
except:
    pass

path = "./mne_data"
os.makedirs(path, exist_ok = True)
path1 = "./RPs"
os.makedirs(path1, exist_ok = True)

def run_computation(t_start, t_end, selected_subject, fir_filter, electrode_name, cut_freq, win_len, n_fft, percentile, run_list, options):
    
    epochs, raw = dataset.eegbci_data(tmin=t_start, tmax=t_end,
                             subject=selected_subject,
                             filter_range=fir_filter,run_list=run_list)

    s_rate = epochs.info['sfreq']

    electrode_index = epochs.ch_names.index(electrode_name)

    electrode_open = epochs.get_data()[0][electrode_index]
    electrode_close = epochs.get_data()[1][electrode_index]

    stft_open = vector_space.compute_stft((electrode_open),
                                        n_fft=n_fft, win_len=win_len,
                                        s_rate=epochs.info['sfreq'],
                                        cut_freq=cut_freq)

    stft_close = vector_space.compute_stft((electrode_close),
                                        n_fft=n_fft, win_len=win_len,
                                        s_rate=epochs.info['sfreq'],
                                        cut_freq=cut_freq)

    # matrix_open = distance_matrix.EuclideanPyRQA_RP_stft(stft_open)
    # matrix_close = distance_matrix.EuclideanPyRQA_RP_stft(stft_close)
    matrix_open = distance_matrix.EuclideanPyRQA_RP_stft_cpu(stft_open)
    matrix_close = distance_matrix.EuclideanPyRQA_RP_stft_cpu(stft_close)

    nbr_open = np.percentile(matrix_open, percentile)
    nbr_close = np.percentile(matrix_close, percentile)

    matrix_open_binary = distance_matrix.set_epsilon(matrix_open,nbr_open)
    matrix_close_binary = distance_matrix.set_epsilon(matrix_close,nbr_close)

    matrix_open_to_plot = matrix_open_binary
    matrix_closed_to_plot = matrix_close_binary

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(16,8),dpi=200)
    ax1.imshow(matrix_open_to_plot, cmap='Greys', origin='lower') #cividis
    ax1.set_xticks(np.linspace(0, matrix_open_to_plot.shape[0] , ax1.get_xticks().shape[0]))
    ax1.set_yticks(np.linspace(0, matrix_open_to_plot.shape[0] , ax1.get_xticks().shape[0]))
    ax1.set_xticklabels([str(np.around(x,decimals=0)) for x in np.linspace(0, matrix_open_to_plot.shape[0] / s_rate, ax1.get_xticks().shape[0])])
    ax1.set_yticklabels([str(np.around(x, decimals=0)) for x in np.linspace(0, matrix_open_to_plot.shape[0] / s_rate, ax1.get_yticks().shape[0])])
    ax1.set_title(options[0]+' window size = 240 samples, ε = '+str(np.round(nbr_open,4)))
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('time (s)')

    ax2.imshow(matrix_closed_to_plot, cmap='Greys', origin='lower')
    ax2.set_xticks(np.linspace(0, matrix_closed_to_plot.shape[0] , ax1.get_xticks().shape[0]))
    ax2.set_yticks(np.linspace(0, matrix_closed_to_plot.shape[0] , ax1.get_xticks().shape[0]))
    ax2.set_xticklabels([str(np.around(x,decimals=0)) for x in np.linspace(0, matrix_closed_to_plot.shape[0] / s_rate, ax1.get_xticks().shape[0])])
    ax2.set_yticklabels([str(np.around(x, decimals=0)) for x in np.linspace(0, matrix_closed_to_plot.shape[0] / s_rate, ax2.get_yticks().shape[0])])
    ax2.set_title(options[1]+' window size = 240 samples, ε = '+str(np.round(nbr_close,4)))
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('time (s)')

    return fig, matrix_open_binary, matrix_close_binary, epochs, stft_open, stft_close


def plot_rqa(matrix_open_binary, matrix_close_binary, min_vert_line_len, min_diagonal_line_len, min_white_vert_line_len,options):

    categories = ['RR', 'DET', 'L', 'Lmax', 'DIV', 'Lentr', 'DET_RR', 'LAM', 'V', 'Vmax', 'Ventr', 'LAM_DET', 'W', 'Wmax', 'Wentr', 'TT']

    result_rqa_open = recurrence_quantification_analysis.get_results(matrix_open_binary,min_vert_line_len, min_diagonal_line_len, min_white_vert_line_len)
    result_rqa_closed = recurrence_quantification_analysis.get_results(matrix_close_binary,min_vert_line_len, min_diagonal_line_len, min_white_vert_line_len) 

    data = pd.DataFrame([result_rqa_open,result_rqa_closed], columns=categories)
    
    data = data.drop(['RR', 'DIV', 'Lmax'],axis=1)
    # print(data)
    min_max_per_variable = data.describe().T[['min', 'max']]
    min_max_per_variable['min'] = min_max_per_variable['min'].apply(lambda x: int(x))
    min_max_per_variable['max'] = min_max_per_variable['max'].apply(lambda x: math.ceil(x))
    # print(min_max_per_variable)


    
    variables = data.columns
    ranges = list(min_max_per_variable.itertuples(index=False, name=None))   

    format_cfg = {
        #'axes_args':{'facecolor':'#84A8CD'},
        'rad_ln_args': {'visible':True, 'linestyle':'dotted'},
        'angle_ln_args':{'linestyle':'dotted'},
        'outer_ring': {'visible':True, 'linestyle':'dotted'},
        'rgrid_tick_lbls_args': {'fontsize':6},
        'theta_tick_lbls': {'fontsize':9, 'backgroundcolor':'#355C7D', 'color':'#FFFFFF'},
        'theta_tick_lbls_pad':3
    }


    fig = plt.figure(figsize=(5,3),dpi=100)
    radar = ComplexRadar(fig, variables, ranges,n_ring_levels=3 ,show_scales=True, format_cfg=format_cfg)


    custom_colors = ['#F67280', '#6C5B7B', '#355C7D']
    k=0
    for g,c in zip(data.index, custom_colors):
        # radar.plot(data.loc[g].values, label=f"condition {g}", color=c, marker='o')
        radar.plot(data.loc[g].values, label=options[k], color=c, marker='o')
        radar.fill(data.loc[g].values, alpha=0.5, color=c)
        k+=1

    radar.use_legend(loc='upper left', bbox_to_anchor=(-0.4, 1.1), fontsize = 'xx-small') #, bbox_to_anchor=(0.15, -0.25),ncol=radar.plot_counter

    return fig

def waterfall_spectrum(stft1, stft2, s_rate, cut_freq, options):

    fig = plt.figure(figsize=(14, 12), dpi=150)
    grid = plt.GridSpec(8, 8, hspace=0.0, wspace=3.5)
    spectrogram1 = fig.add_subplot(grid[0:3, 0:4])
    spectrogram2 = fig.add_subplot(grid[0:3, 4:])

    spectrogram1.pcolormesh(stft1.T,cmap='viridis')
    spectrogram1.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft1.shape[0], 5)))
    spectrogram1.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, stft1.shape[0] / s_rate, spectrogram1.get_xticks().shape[0])])
    spectrogram1.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft1.shape[1], 5)))
    spectrogram1.set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 5)])
    spectrogram1.set_ylabel('Freq (Hz)', )
    spectrogram1.set_xlabel('Time (s)', )
    spectrogram1.set_title(options[0] + ' Spectrogram', )
    
    spectrogram2.pcolormesh(stft2.T,cmap='viridis')
    spectrogram2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft2.shape[0], 5)))
    spectrogram2.set_xticklabels([str(np.round(x, 1)) for x in np.linspace(0, stft2.shape[0] / s_rate, spectrogram2.get_xticks().shape[0])])
    spectrogram2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.linspace(0, stft2.shape[1], 5)))
    spectrogram2.set_yticklabels([str(np.round(x, 1)) for x in np.linspace(0, cut_freq, 5)])
    spectrogram2.set_ylabel('Freq (Hz)', )
    spectrogram2.set_xlabel('Time (s)', )
    spectrogram2.set_title(options[1] +' Spectrogram', )
    return fig

def save(matrix_open_binary, matrix_close_binary):
    
    file_name_open = './RPs/subject-'+str(selected_subject)+'_electrode-'+electrode_name+'_percentile-'+str(percentile)+'_run-open_binary.npy'
    np.save(file_name_open, np.asarray(matrix_close_binary, dtype=np.ubyte))
    file_name_close = './RPs/subject-'+str(selected_subject)+'_electrode-'+electrode_name+'_percentile-'+str(percentile)+'_run-close_binary.npy'
    np.save(file_name_close, np.asarray(matrix_close_binary, dtype=np.ubyte))

def download():

    file_paths = glob('./RPs/*')

    with ZipFile('download.zip','w') as zip:
        for file in file_paths:
            # writing each file one by one
            zip.write(file)

    return open('download.zip', 'rb')

# ---------------Settings--------------------

st.set_page_config(layout="wide")
st.title('BrainPulse Playground')
sidebar = st.sidebar

selected_subject = sidebar.slider('Select Subject', 0, 100, 25)

electrode_name = sidebar.selectbox(
    'Select Electrode',
    ('FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'))

t_start, t_end = sidebar.slider(
    'Select a time range in seconds',
    0.0, 60.0, (0.0, 30.0))

f1, f2 = sidebar.slider(
    'Select a FIR filter range',
    0.0, 60.0, (2.0, 50.0))
fir_filter = [f1, f2]

cut_freq = f2

win_len = sidebar.slider('FFT window size', 0, 512, 170)

n_fft = sidebar.slider('numer of FFT bins', 0, 1024, 512)

min_vert_line_len = sidebar.slider('Minimum  vertical line length', 0, 250, 2) 

min_diagonal_line_len = sidebar.slider('Minimum diagonal line length', 0, 250, 2) 

min_white_vert_line_len = sidebar.slider('Minimum white vertical line length', 0, 250, 2)

percentile = sidebar.slider('Precentile', 0, 100, 24) 

sidebar.download_button('Download file', download(),file_name='archive.zip')

# ---------------Plot RPs--------------------
runs_ = ['Baseline open eyes', 'Baseline closed eyes', 'Motor execution: left vs right hand', 'Motor imagery: left vs right hand',
    'Motor execution: hands vs feet', 'Motor imagery: hands vs feet']

options = st.multiselect('Select two runs to compare', runs_, ['Baseline open eyes', 'Baseline closed eyes'])

run_list = []

for v in options:
    run_list.append(runs_.index(v)+1)
if len(run_list) <= 1:
    run_list = [1,2]
    
rp_plot, matrix_open_binary, matrix_close_binary, epochs, stft1, stft2  = run_computation(t_start, t_end, selected_subject, fir_filter, electrode_name, cut_freq, win_len, n_fft, percentile, run_list,options)
st.write(rp_plot)

# ---------------Plot Spectrum-------------------- 
st.write(waterfall_spectrum(stft1, stft2, 160, cut_freq, options))

# ---------------Save RPs--------------------
if st.button('Save RPs as *.npy'):
    save(matrix_open_binary, matrix_close_binary)

# ---------------Plot Radar--------------------
rqa_radar = plot_rqa(matrix_open_binary, matrix_close_binary, min_vert_line_len, min_diagonal_line_len, min_white_vert_line_len, options)
st.write(rqa_radar)




