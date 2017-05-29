'''
@author: ahmed allam <ahmed.allam@yale.edu>

Some parts of the script are adapted from `sussexwearlab <https://github.com/sussexwearlab/DeepConvLSTM/blob/master/preprocess_data.py>`__ 
'''

import os
import subprocess
import numpy as np
import pandas as pd
from pyseqlab.utilities import create_directory

current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
dataset_dir = os.path.join(root_dir, 'dataset')

"""
1-Download the opportunity dataset from https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition#
2-Place it under the dataset directory in this repository
"""
# directory to the downloaded dataset
oppdataset_dir = os.path.join(dataset_dir, 'OpportunityUCIDataset', 'dataset')

# identify training and testing files
training_files = []
for subject in ('S1', 'S2', 'S3'):
    if(subject != 'S1'):
        r_val = range(1, 4)
    else:
        r_val = range(1,6)
    for i in r_val:
        training_files.append(os.path.join(oppdataset_dir, "{0}-ADL{1}.dat".format(subject, i)))
training_files += [os.path.join(oppdataset_dir, subject+'-Drill.dat') for subject in ('S1', 'S2', 'S3')]    
test_files = [os.path.join(oppdataset_dir, "{0}-ADL{1}.dat".format(subject, i)) for i in range(4, 6) for subject in ('S2', 'S3')]

def select_columns_opp(data):
    """select the 113 columns employed in the OPPORTUNITY challenge
    
    Args:
        data: numpy array representing opportunity data file (i.e. sensor measurements and labels)
    
    Return: 
        numpy array representing the specified/selected columns
    """

    features_delete = np.arange(46, 50)
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])
    return(np.delete(data, features_delete, 1))

def normalize(data, method = None):
    """normalize sensor channels
      
      Args:
          data: numpy array representing the sensor measurements
        
      Return:
          Normalized sensor data
    """
    try:
        if(method == 'rescaling'):
            data_normalized = (data - data.min(axis=0))/(data.ptp(axis=0))
        elif(method == 'standardized'):
            data_normalized = (data - data.mean(axis=0))/(data.std(axis=0))
    except Exception as e:
        print("dividing by zero exception.. Some of the features have either a range (max - min) equal to 0 or the standard deviation is 0 (i.e. the feature is constant)..")
        print(e)
    return(data_normalized)


def separate_x_y(data):
    """separate the data samples into features (X) and labels (Y)

    Args:
        data: numpy array representing the data read from the opportunity dataset

    Return:
         tuple(numpy array, numpy array, numpy array) representing the X sensor data and Y labels for locomotion and gestures
    """

    data_x = data[:, 1:114]
    data_loc = data[:, 114]  # Locomotion label
    data_gest = data[:, 115]  # Gestures label

    return(data_x, data_loc, data_gest)


def adjust_y_labels(data_y, label='locomotion'):
    """transform original labels into the range [0, nb_labels-1]

    Args:
        data_y: numpy array representing sensor labels
        label: string, ['locomotion' (default), 'gestures']
               Type of activities to be recognized
    Return:
        numpy array representing the modified sensor labels (i.e. Y)
    """

    if(label == 'locomotion'):  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif(label == 'gestures'):  # Labels for gestures are adjusted
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return(data_y)


def process_dataset_file(data, interp_type='linear', norm_type = None):
    """process individual OPPORTUNITY files

    Args:
        data: numpy matrix representing the loaded dataset file 
              containing data samples (rows) for every sensor channel (column)

    Keyword args:
        interp_type: string, {'linear'}
        norm_type: None (default), or string in {'rescaling', 'standardized'}
                   decide either rescaling or standardization or None
    Return:
        tuple(numpy array, dict) representing the X sensor data and Y labels for locomotion and gestures
    """

    # Select correct columns based on the opportunity challenge
    data = select_columns_opp(data)

    # data is separated into features and labels
    data_x, data_loc, data_gest =  separate_x_y(data)
    l = {'locomotion':data_loc, 'gestures':data_gest}
    for label in l:
        l[label] = adjust_y_labels(l[label], label)
        l[label] = l[label].astype(int)

    # Perform linear interpolation
    if(interp_type == 'linear'):
        data_x = np.array([pd.Series(i).interpolate() for i in data_x.T]).T

    # Remaining missing data is converted to zero
    data_x[np.isnan(data_x)] = 0

    if(norm_type): 
        # sensor channels are normalized/standardized
        data_x = normalize(data_x, method=norm_type)

    return(data_x, l)

def get_window_label(indices_info, df_y):
    """choose/decide the label for the sliding window
    
    Args:
        indices_info: dictionary containing the start_indx (optional) and stop_indx of the sliding windows
        df_y: pandas data frame that represents the Y labels. It has a dimension of nx1
    """
    start_windx = indices_info['start_windx']
    stop_windx = indices_info['stop_windx']
    cols = df_y.columns
    # case of mode (i.e. choose the most frequent label)
    if(start_windx):
        y_labels = []
        for i in range(len(start_windx)):
            y_labels.append(df_y[cols[-1]].iloc[start_windx[i]:stop_windx[i]].value_counts().index.tolist()[0])
        df_y_proc = pd.DataFrame(y_labels, columns = cols.tolist())
    
    # case of getting the label of the last measurement in the window
    else:
        df_y_proc = df_y.loc[stop_windx, cols.tolist()].copy()
    # it could be also int8
    df_y_proc = df_y_proc.astype('int32')
    return(df_y_proc)


def generate_data(data_files, data_type, **kwargs):
    """read the OPPORTUNITY challenge raw data and process all sensor channels

    Args:
        data_files: list of file paths to the opportunity dataset. This list could be ``training_files`` or ``testing_files``
        data_type: string, {'train', 'test'}
    
    Keyword Args:
        wsize: integer, the sliding window size to segment the sensor measurements
        step_size: integer, step size between two consecutive sliding windows
        y_agg_method: string, {'last', 'mode'}. 
                      If ``last`` is chosen then the label for the window will be the label of the last measurement in the window. 
                      else if ``mode`` is chosen, then the most/frequent occurring label will be chosen for the window.
        func_type: tuple, containing name of functions to be applied on windows. 
                   Possible values {'mean', 'std', 'sum'}
    """
    num_sensors_channels = 113
    # assign short names for the sensor readings
    sensor_names = ["s_{}".format(i) for i in range(num_sensors_channels)] 
    y_options = ('locomotion', 'gestures')
    num_yoptions = len(y_options)
    # parse keyword arguments (kwargs)
    wsize = kwargs.get('wsize')
    if(not wsize):
        wsize = 0
        step_size = 0
        y_agg_method = None
        func_type = None
    else:
        step_size = kwargs.get('step_size')
        if(not step_size):
            # step size will be half the window size
            step_size = int(np.ceil(wsize/2))
        y_agg_method = kwargs.get('y_agg_method')
        if(not y_agg_method):
            y_agg_method = 'last'
        func_type = kwargs.get('func_type')
        if(not func_type):
            func_type = ('mean',)
    print('Processing dataset files ...')
    out_files = []
    for option in y_options:
        option_dir = create_directory(option, dataset_dir)
        outfile = os.path.join(option_dir, "sensor_{}_wsize{}_stepsize{}_yagg{}_{}".format(data_type, 
                                                                                           wsize,
                                                                                           step_size,
                                                                                           y_agg_method, 
                                                                                           option))
        out_files.append(outfile)
    agg_dfs = [pd.DataFrame(), pd.DataFrame()]
    session_counter = 1
    for data_file in data_files:
        data_x = np.empty((0, num_sensors_channels))
        data_loc = np.empty((0))
        data_gest = np.empty((0))
        try:
            data = np.loadtxt(data_file)
            fname = os.path.basename(data_file)
            print('... file {0}'.format(fname))
            x, l = process_dataset_file(data)
            data_x = np.vstack((data_x, x))
            data_loc = np.concatenate([data_loc, l['locomotion']])
            data_gest = np.concatenate([data_gest, l['gestures']])
            # transform all data into pandas data frame
            data_x = pd.DataFrame(data_x, columns=sensor_names)
            data_loc = pd.DataFrame(data_loc, columns=['locomotion'])
            data_gest = pd.DataFrame(data_gest, columns=['gestures'])
            if(wsize):
                # apply the sliding window approach for segmenting the sensor measurements
                data_x_sw = pd.DataFrame()
                for functype in func_type:
                    data_x_sw = pd.concat([data_x_sw, slide_window_df(data_x, wsize, step_size, functype)], axis=1)
                data_x = data_x_sw    
                stop_indx_arr = data_x.index.values
                stop_indx = stop_indx_arr.tolist()
                if(y_agg_method == 'last'):
                    indices_info = {'start_windx':None,'stop_windx':stop_indx}
                elif(y_agg_method == 'mode'):
                    start_indx_arr = stop_indx_arr - (wsize - 1)
                    start_indx_arr[-1] = start_indx_arr[-2] + step_size
                    indices_info = {'start_windx':start_indx_arr.tolist(),'stop_windx':stop_indx}
                # get the label corresponding to the window based on the method specified (i.e. 'last' or 'mode')
                data_loc = get_window_label(indices_info, data_loc)
                data_gest = get_window_label(indices_info, data_gest)
                data_x.columns = sensor_names
            data_loc = data_loc.astype('int32')
            data_gest = data_gest.astype('int32')
            print("data (sensors) shape: ", data_x.shape)
            # join the data frames and write all as one sequence
            target_dfs = []
            for df_y in [data_loc, data_gest]:
                curr_df = pd.concat([data_x, df_y], axis=1)
                curr_df['seq_id'] = session_counter
                target_dfs.append(curr_df)
            for i in range(num_yoptions):
                agg_dfs[i] = pd.concat([agg_dfs[i], target_dfs[i]], axis=0, ignore_index=True)
                print("confirming the shape..")
                print("y choice {}".format(y_options[i]))
                print("agg_df.shape = {}".format(agg_dfs[i].shape))
            session_counter+=1
        except Exception as e:
            print('something went wrong !!!')
            print(e)

    # pickle data frames
    for i, y_option in enumerate(y_options):
        agg_dfs[i].to_pickle(out_files[i]+".pkl")
        # write data frames as sequence files
        option_dir = create_directory(y_option, dataset_dir)
        track_colnames = sensor_names + [y_option] 
        write_seqs_tofile(agg_dfs[i], os.path.join(option_dir, '{}.txt'.format(data_type)), track_colnames, sep = " ", header=False)
    # return a tuple of (locomotion, gestures data)
    return(agg_dfs[0], agg_dfs[1])
   
     
def run_data_generation_workflow(discretize=True, iob_repr=True):
    data_options = (('train', training_files),('test', test_files))
    wsize = 15
    for elem in data_options:
        dtype, dfiles = elem
        __, __ = generate_data(dfiles, dtype, wsize=wsize)
        if(iob_repr or discretize):
            for y_option in ('locomotion', 'gestures'):
                option_dir = os.path.join(dataset_dir, y_option)
                if(discretize):
                    discretize4crf('{}.txt'.format(dtype), dtype, option_dir, '{}_discretized.txt'.format(dtype))
                    target_fname = '{}_discretized'.format(dtype)
                else:
                    target_fname = '{}'.format(dtype)
                if(iob_repr):
                    generate_iob_repr_file("{}.txt".format(target_fname), option_dir, '{}_iob.txt'.format(target_fname), " ", 0)

def discretize4crf(input_filename, input_type, input_dir, out_filename):
    """ discretize continuous data using MDLPC algorithm
            
        Args:
            input_filename: string, input file name with extension  
                            The file would contain the continuous features where each feature represents a column
                            each separated by tab or space. The final column is reserved for the y (label value) 
            input_type: string, {'train' or 'test'}
            input_dir: string, path to the directory where the input file resides
            out_filename: string, name with extension for the generated output file (i.e. file containing the discretized data)
                       
        .. warning:
        
           the `discretize4crf <https://gforge.inria.fr/projects/discretize4crf/>`__ should be already installed in the system.
            
    """
    # execute the shell command
    out_filepath = os.path.join(input_dir, out_filename)
    f_out = open(out_filepath, 'a')
    if(input_type == 'train'):
        command = "discretize4crf -train {} -to {} -save {} -oe".format(os.path.join(input_dir, input_filename),
                                                                        out_filepath,
                                                                        os.path.join(input_dir, "train_intervals.txt"))

    elif(input_type == 'test'):
        command = "discretize4crf -test {} -eo {} -load {} -oe".format(os.path.join(input_dir, input_filename),
                                                                       out_filepath,
                                                                       os.path.join(input_dir, "train_intervals.txt"))
    command_lst = command.split(" ")
    print(command_lst)
    try:
        process = subprocess.Popen(command_lst, stdout=f_out)
        output, err = process.communicate()
        print("Finished discretizing data in {}".format(input_filename))
        f_out.close()
    except Exception as e:
        print("something went wrong!!")
        print(e)
        
def slide_window_df(df, win_size, step_size, func_type = "mean"):
    suffix = func_type
    if(func_type == 'mean'):
        m_df = df.rolling(win_size).mean()
    elif(func_type == 'sum'):
        m_df = df.rolling(win_size).sum()
    elif(func_type == 'std'):
        m_df = df.rolling(win_size).std()
    m_df = m_df[win_size-1::step_size]
    last_index=m_df.index.tolist()[-1]
    # we need to compute the last window
    df_last_index = df.shape[0]
    if(last_index<df_last_index):
        start_indx = df_last_index-1-win_size+step_size-1
        if(func_type == 'mean'):
            l_df = df.iloc[start_indx:].rolling(df_last_index-start_indx).mean()
        elif(func_type == 'sum'):
            l_df = df.iloc[start_indx:].rolling(df_last_index-start_indx).sum()
        elif(func_type == 'std'):
            l_df = df.iloc[start_indx:].rolling(df_last_index-start_indx).std()       
        l_df.dropna(axis=0, how='all', inplace=True)
        m_df = pd.concat([m_df, l_df], axis=0)
    m_df.columns = ["{}_{}".format(col, suffix) for col in m_df.columns.tolist()]        
    return(m_df)
    
def write_seqs_tofile(df, fpath, cols, sep="\t", header=True):
    if(header):
        f_out = open(fpath, 'a')
        f_out.write(sep.join(cols) + "\n")
        f_out.close()
    grouped_df = df.groupby(['seq_id'])
    total_seqs = len(grouped_df)
    print("number of sequences is: ", total_seqs)
    for __, group_seq in grouped_df:
        group_seq.to_csv(fpath, mode='a', columns=cols, index=False, header=False, sep=sep, na_rep='NaN')
        f_out = open(fpath, 'a')
        f_out.write("\n")
        f_out.close()
        total_seqs -= 1
        print("{} sequences left".format(total_seqs))

def generate_iob_repr_file(input_filename, input_dir, out_filename, col_sep, other_symbol):
    other_symbol = str(other_symbol)
    fout = open(os.path.join(input_dir, out_filename), 'a')
    with open(os.path.join(input_dir, input_filename)) as file_obj:
        counter = 0
        y_stack = []
        for line in file_obj:
            counter += 1
            line = line.rstrip()
            if line:
                *x_arg, y = line.split(col_sep)
                if(y == other_symbol):
                    iob_repr = y
                else:
                    if(y_stack):
                        if(y_stack[-1]==y):
                            iob_repr = 'I-'+y
                        else:
                            iob_repr = 'B-'+y
                    else:
                        iob_repr = 'B-'+y
                y_stack.append(y)
                fout.write(col_sep.join(x_arg) + col_sep + iob_repr + "\n")
            else:
                y_stack = []
                fout.write("\n")
    fout.close()
                
    
def generate_iob_repr_df(df, target_col, other_symbol):
    grouped_df = df.groupby(['seq_id'])
    total_seqs = len(grouped_df)
    print("number of sequences is: ", total_seqs)
    iob_rep_df = pd.DataFrame()
    #all_cols = df.columns.tolist()
    for __, ggroup in grouped_df:
        g_df = pd.DataFrame()
        iob_rep = []
        track_elem = []
        for i in range(ggroup.shape[0]):
            elem = ggroup[target_col].iloc[i]
            if(elem != other_symbol):
                if(iob_rep):
                    if(track_elem[-1] == elem):
                        iob_rep.append("I-{}".format(elem))
                    else:
                        iob_rep.append("B-{}".format(elem))
                else:
                    iob_rep.append("B-{}".format(elem))
            else:
                iob_rep.append("{}".format(elem))
            track_elem.append(elem) 
        g_df[target_col+"_iob"] = iob_rep
        #g_df['seq_id'] = gkey
        ggroup_df = pd.DataFrame(ggroup)
        ggroup_df.reset_index(inplace=True, drop=True)
        g_df = pd.concat([ggroup_df, g_df], axis=1, ignore_index=True)
        iob_rep_df = pd.concat([iob_rep_df, g_df], axis=0, ignore_index=True)       
        total_seqs -= 1
        print("{} sequences left".format(total_seqs))
    iob_rep_df.columns = df.columns.tolist() + [target_col+"_iob"]
    return(iob_rep_df)

if __name__ == '__main__':
    pass