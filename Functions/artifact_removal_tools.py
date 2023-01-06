"""
    Artifact Removal Tools
    -----------------
    These functions are used to remove artifacts from preprocessed EEG data

"""

## Import libraries
import os
import numpy as np
import scipy.stats as stats
import scipy.linalg as linalg
from sklearn.cluster import KMeans
from numpy import matlib as matlib, ndarray
import concurrent.futures as cf

def remove_eyeblinks_cpu(eeg_raw, srate, analysis='offline', window_length = 125, n_clusters = 4, fd_threshold = 1.4, ssa_threshold = 0.01, svd_method='sci', antidiag_method='mask', enable_multithread=False):    
    """
        This function removes eyeblink artifacts from EEG data using an SSA approach implementation was adapted from
        `[Maddirala & Veluvolu 2021]` 

    Parameters
    ----------
        eeg_raw: array_like 
            Raw EEG data to be cleaned
        srate: int or float
            Sampling rate of the raw EEG signal [Hz]
        analysis: string, optional
            Type of analysis to perform. Size of dimensions should be in the following order:
            'offline': n_channels < n_epochs < n_samples (default)
            'online': n_epochs < n_channels < n_samples
        window_lenght: float or int, optional
            Length of window used to create the SSA matrix 
            If window_length.type() == float : window_length must be in msec 
            If window_length.type() == int: window_length is the number of sample points
        fd_threshold: float, optional
            Fractal dimension threshold 
        ssa_threshold: floar, optional
            Singular Spectrum Analysis threshold
        svd_method: str, optional
            Method to use for SVD calculation
            'sci': Scipy (default)
            'np': Numpy
        antidiag_method: str
            Method used to calculate average of antidiagonals
            'simple': Iterate through each antidiagonal and calculate mean value
            'mask': Compute matrix with antidiagonals and boolean mask, calculate mean of matrix
        enable_multithread: bool
            Enable multithreading processing in CPU. Uses half of the available logical processors

    Returns
    -------
        eeg_clean: array like
            EEG signal without the artifacts
        eeg_artifact: array like
            Artifacts found in EEG signal
        saturation_found: boolean
            Boolean to determine wether the signal hit one of the rails (clipping)

    Notes
    -----
        Return values have the same shape as eeg_raw
    """

    #%% Organize data
    # - Determine if number of dimensions of data
    shape = np.shape(eeg_raw)
    dimensions = np.ndim(eeg_raw)
    data_reshape = False    # Initialize flag to know if the data was reshaped

    if dimensions < 3:
        if dimensions == 1:     # If data is one dimension, expand to row matrix
            eeg_raw = np.reshape(eeg_raw, (1,-1))
        elif dimensions == 2:   # If data is a matrix, Make sure data is in row vectors (i.e., [channels x samples])
            if shape[0] > shape[1]:
                eeg_raw = eeg_raw.T
                data_reshape = True 
        
        [eeg_clean, artifact_found, saturation_found] = remove_eyeblinks_2D(eeg_raw, srate, data_reshape, window_length = window_length, n_clusters = n_clusters,
                                                                            fd_threshold = fd_threshold, ssa_threshold = ssa_threshold, svd_method=svd_method, 
                                                                            antidiag_method=antidiag_method, enable_multithread=enable_multithread)

    elif dimensions == 3:   # If data is a tensor, make sure data is in the right shape according to analysis_type[analysis]
        analysis_type = {'online':np.array([0,1,2]), 'offline':np.array([1,0,2])}
        shape_order = np.argsort(shape)

        # If the order is not correct, transpose to correct order
        if not np.array_equiv(shape_order, analysis_type[analysis]):
            eeg_raw = eeg_raw.transpose(shape_order[analysis_type[analysis]])

        # Preallocate variables
        shape = np.shape(eeg_raw) # Shape [epoch x channels x samples]
        eeg_clean = np.zeros_like(eeg_raw)
        artifact_found = np.zeros((shape[0], shape[1]))
        saturation_found = np.zeros((shape[0], shape[1]))

        # Run eye_blink removal for each epoch
        for e in range(shape[0]):
            [eeg_clean[e,:,:], artifact_found[e,:], saturation_found[e,:]] = remove_eyeblinks_2D(eeg_raw[e,:,:], srate, data_reshape, window_length = window_length, n_clusters = n_clusters,
                                                                                fd_threshold = fd_threshold, ssa_threshold = ssa_threshold, svd_method=svd_method, 
                                                                                antidiag_method=antidiag_method, enable_multithread=enable_multithread)
    else:
        print("Warning, data dimension not accepted")
        return None

    return eeg_clean, artifact_found, saturation_found
    

def remove_eyeblinks_2D(eeg_raw, srate, data_reshape, window_length = 125, n_clusters = 4, fd_threshold = 1.4, ssa_threshold = 0.01, svd_method='sci', antidiag_method='mask', enable_multithread=False):
    """
        This function implements the artifact removal tool in 2D matrices


    """

    # - Determine embedding matrix sizes
    shape = np.shape(eeg_raw)   # Update shape value
    n_channels = shape[0]       # Number of channels 
    N = shape[1]                # Length of EEG signal
    
    #%% Create embedding matrix
    # - M = Number of rows
    if type(window_length) == int:
        M = window_length
    elif type(window_length) == float:
        M = np.floor(window_length * srate).astype(int) 
    else: 
        print('Data type of window_length is incorrect \n Data type should be "int" or "float"')
        return None

    # - K = Number of columns
    K = N - M + 1

    # - Create embedding matrix with the correspongding indices of the vector data
    idx_col = np.arange(K).reshape(1,-1).repeat(M, axis=0)
    idx_row = np.arange(M).reshape(-1,1).repeat(K, axis=1)
    idx_mat = idx_col + idx_row

    #%% Decomposed 
    # - Preallocate variables
    eeg_clean = np.zeros_like(eeg_raw)           # Artifact signal (after SSA)
    artifact_found = np.zeros(n_channels)
    saturation_found = np.zeros(n_channels)

    #%% Run Artifact removal in each channel
    # - Multithreaded enabled
    if enable_multithread:
        [eeg_clean, _] = multithread(eeg_raw, idx_mat, n_clusters, fd_threshold, ssa_threshold, svd_method, antidiag_method)

    # - Multithread disabled
    else:
        for channel in range(n_channels):
            [eeg_clean[channel,:], artifact_found[channel], saturation_found[channel]] = single_remove_eyeblinks(eeg_raw=eeg_raw[channel,:], 
            idx_mat=idx_mat, svd_method=svd_method, antidiag_method=antidiag_method)

    #%% Return data in original shape
    if data_reshape:
        return eeg_clean.T, artifact_found, saturation_found
    else:
        return eeg_clean, artifact_found, saturation_found

def single_remove_eyeblinks(eeg_raw, idx_mat, n_clusters = 4, fd_threshold = 1.4, ssa_threshold = 0.01, var_tol=1e-15, svd_method = 'sci', antidiag_method = 'mask'):
    """
        This function implements the artifact removal described in [Maddirala & Veluvolu 2021].

    Parameters
    ----------
        eeg_raw: array_like 
            Single channel raw EEG data to be cleaned
        idx_mat: array_like
            Matrix with the indices to build the embedded matrix.
        n_clusters: int, optional
            Number of clusters to use in the kmeans classifier
        fd_threshold: float, optional
            Fractal dimension threshold 
        ssa_threshold: float, optional
            Singular Spectrum Analysis threshold
        svd_method: str, optional
            Method to use for SVD calculation
            'sci': Scipy
            'np': Numpy
        method: str
            Method used to calculate average of antidiagonals
            'mask': Compute matrix with antidiagonals and boolean mask, calculate mean of matrix
            'simple': Iterate through each antidiagonal and calculate mean value            

    Returns
    -------
        artifact: array like
            Single artifacts vector found in EEG signal
        artifact_found: boolean
            Boolean to determine wether there were artifacts detected
        saturation_found: boolean
            Boolean to determine wether the signal hit one of the rails (clipping)
    """

    #%% Create EEG embedded matrix from single row of EEG
    eeg_embedded = eeg_raw[idx_mat]

    #%% Determine number of samples, rows, and columns
    N = np.size(eeg_raw,0)  # Number of samples [N]
    M = np.size(idx_mat,0)  # Number of rows in index matrix [N]
    K = np.size(idx_mat,1)  # Number of columns in index matrix [N]
    
    #%% Calculate features from embedded matrix
    f1 = (eeg_embedded**2).sum(axis=0)              # Energy [V^2]
    f2 = np.sqrt((np.diff(eeg_embedded,axis=0)).var(axis=0)) / eeg_embedded.var(axis=0) # H_mobility
    f3 = stats.kurtosis(eeg_embedded, axis=0)       # Kurtosis
    f4 = eeg_embedded.max(0) - eeg_embedded.min(0)  # Range
    eeg_features = np.array((f1,f2,f3,f4))

    #%% Replace NaN values
    # - This can happen when the data clips, usually the variance will be 0
    #   this can cause issues when performing the kmeans classification
    # - If this happens, the window is unusable for further classification. Return 0 data 
    if np.any(eeg_embedded.var(axis=0) < var_tol):
    # if np.any(np.isnan(eeg_features)):
       artifact_found = False
       saturation_found = True
       return np.zeros_like(eeg_raw), artifact_found, saturation_found

    #%% Perform Kmeans classification
    kmeans = KMeans(n_clusters=n_clusters).fit(eeg_features.T)

    #%% Compute decomposed matrices
    # - Preallocate variables
    eeg_component = np.zeros((n_clusters, N))

    # - Calculate EEG component for each cluster
    for cluster in range(n_clusters):
        eeg_decomposed = np.zeros((M,K))    # Temporary matrix to store the decomposed EEG for each cluster [M,K] 
        
        # - Determine columns to copy based on the kmeans label
        copy_index = (kmeans.labels_==cluster) 
        eeg_decomposed[:,copy_index] = eeg_embedded[:,copy_index]
                
        # Reconstruct signal from antidiagonal averages
        eeg_component[cluster, :] = mean_antidiag(eeg_decomposed, antidiag_method)
        
    #%% Fractal Dimension (FD)        
    # - Normalize EEG to unit square
    x_norm = np.repeat(np.reshape(np.linspace(0, 1, N),[-1,1]),n_clusters,axis=1)
    y_num = eeg_component - matlib.repmat(np.amin(eeg_component, axis=1, keepdims=True), 1, N)
    y_den = matlib.repmat(np.amax(eeg_component, axis=1, keepdims=True) - np.amin(eeg_component, axis=1, keepdims=True), 1, N)
    y_norm = np.divide(y_num, y_den).T
    z = np.array([x_norm, y_norm]) # 3D Matrix to store x_norm and y_norm [sample x [x,y] x n_cluster]

    # - Calculate fractal dimension
    # l = np.sum(np.sum(np.abs(np.diff(z, axis=1)), axis=0), axis=0)  # Calculate length of signal (l1-norm for each n_cluster) [A.U.]
    l = np.sum(np.sqrt(np.sum(np.square(np.diff(z, axis=1)), axis=0)), axis=0)  # Calculate length of signal (l2-norm for each n_cluster) [A.U.]
    fd = 1 + (np.log(l) / np.log(2*(N-1)))  # Fractal dimension value

    # - Binary artifact creation
    fd_mask = fd < fd_threshold                         # Apply threshold to FD to determine artifact components

    #&& No artifacts found
    # - Return original data
    if not fd_mask.any():
        artifact_found = False
        saturation_found = False
        return eeg_raw, artifact_found, saturation_found

    artifact_found = True                               # Flag to know that artifacts were found
    saturation_found = False                            # Flag to know the channel hit the rail
    eeg_mask = np.sum(eeg_component[fd_mask,:],0)       # Vector with artifact points != 0
    eeg_artifact = eeg_raw*(eeg_mask != 0).astype(int)  # Multiply mask with original to get eeg_artifact with correct values [V]

    #%% Singular Spectrum Analysis
    # - Singular Value Decomposition
    artifact_embedded = eeg_artifact[idx_mat]       # Create multivariate matrix for each channel
    
    # - Use scipy or numpy for SVD calculation
    if svd_method == 'sci':
        [u, s, vh] = linalg.svd(artifact_embedded, full_matrices=False)
        pass
    elif svd_method == 'np':
        [u, s, vh] = np.linalg.svd(artifact_embedded, full_matrices=False)
    else:
        print("wrong SVD method selected")
        return None

    # - Determine number of groups
    eigen_ratio = (s / s.sum()) > ssa_threshold # Keep only eigenvectors > ssa_threshold
    # vh_sub = vh[0:s.size]                       # Select subset of unitary arrays
    # artifact_sub = u[:,eigen_ratio] @ np.diag(s[eigen_ratio]) @ vh_sub[eigen_ratio,:]   # Artifact with subset of eigenvectors
    artifact_sub = u[:,eigen_ratio] @ np.diag(s[eigen_ratio]) @ vh[eigen_ratio,:]   # Artifact with subset of eigenvectors

    # Reconstruct signals from antidiagonal averages
    artifact = mean_antidiag(artifact_sub, antidiag_method)

    eeg_clean = eeg_raw - artifact

    return eeg_clean, artifact_found, saturation_found

def mean_antidiag(input_mat, method):
    """
        This function returns the mean of the antidiagonal components of a matrix

        Parameters
        ----------
            input_mat: array_like
                Matrix with shape [i,k] for which the mean of the antidiagonal components will be calculated.\n
                Must be a 2D matrix.
            method: str
                Method used to calculate average of antidiagonals
                'mask': Compute matrix with antidiagonals and boolean mask, calculate mean of matrix
                'simple': Iterate through each antidiagonal and calculate mean value
                
        Returns
        -------
            average_vect: array_like
                1D vector containing the mean values of the antidiagonal components
    """

    input_shape = input_mat.shape       # Shape of input matrix
    input_flip = np.fliplr(input_mat)   # Flip input matrix from left to right
    n = np.sum(input_shape) - 1         # Number of samples of resulting vector 

    # Make sure that input matrix is 2D
    if len(input_shape)!=2:
        print('Matrix must be 2D')
        return None

    # Calculate mean across diagonals
    if method == 'simple':
        average_vect = np.zeros(n)  # Preallocate vector with average values

        for i_diag, k_diag in enumerate(range(-input_shape[0]+1,input_shape[1])):
            average_vect[-i_diag-1] = input_flip.diagonal(offset=k_diag).mean() # Organize values from end to start to get right order

    elif method == 'mask':
        max_diag = (input_flip.diagonal(offset=0)).size # Size of longest diagonal
        diag_mat = np.zeros((max_diag,n))               # Empty matrix to store antidiagonal values
        mask_mat = np.zeros((max_diag,n))               # Empty matrix to store mask values

        for i, k_diag in enumerate(range(-input_shape[0]+1, input_shape[1])):
            diag_vals = input_flip.diagonal(offset=k_diag)  # Values of the k^th diagonal
            n_diag = diag_vals.size                         # Length of values of the k^th diagonal
            diag_mat[0:n_diag,i] = diag_vals
            mask_mat[0:n_diag,i] = 1

        average_vect = np.flip(diag_mat.mean(axis=0, where=mask_mat.astype(bool)))

    else:
        print('Antidiagonal method not available')
        return None

    return average_vect

def multithread(eeg_raw, idx_mat, n_clusters, fd_threshold, ssa_threshold, svd_method, antidiag_method):
    """
    This function calls the single_remove_eyeblinks function and parallelizes the code in multiple threads

    Parameters
    ----------
        eeg_raw: array_like 
            2D multiple channel raw EEG data to be cleaned [channels, samples]
        idx_mat: array_like
            Matrix with the indices to build the embedded matrix.
        n_clusters: int, optional
            Number of clusters to use in the kmeans classifier
        fd_threshold: float, optional
            Fractal dimension threshold 
        ssa_threshold: float, optional
            Singular Spectrum Analysis threshold
        svd_method: str, optional
            Method to use for SVD calculation
            'sci': Scipy
            'np': Numpy
        method: str
            Method used to calculate average of antidiagonals
            'mask': Compute matrix with antidiagonals and boolean mask, calculate mean of matrix
            'simple': Iterate through each antidiagonal and calculate mean value     

    Returns
    -------
        average_vect: array_like
            1D vector containing the mean values of the antidiagonal components

    Notes
    -----
        The paralellization is affected by Python's GIL, there might not be any speed benefits of using this function.
    """
    #%% Setup
    # - Determine number of channels    
    n_channels = np.size(eeg_raw, axis=0)

    # - Preallocate variables
    artifact = np.zeros_like(eeg_raw)
    eeg_list = [None] * n_channels
    idx_mat_list = [None] * n_channels
    
    # - Organize input variables as lists
    for i in range(n_channels):
        eeg_list[i] = eeg_raw[i,:]
        idx_mat_list[i] = idx_mat

    svd_list = [svd_method] * n_channels
    antidiag_list = [antidiag_method] * n_channels
    n_clusters_list = [n_clusters] * n_channels
    fd_threshold_list = [fd_threshold] * n_channels
    ssa_threshold_list = [ssa_threshold] * n_channels

    # Use ThreadPoolExecutor to parallelize the code
    # - Determine number of CPUs (threads)
    total_cpus = os.cpu_count()
    n_workers = np.floor(total_cpus/2).astype(int)
    
    i = 0   # Initialize counter for channel
    with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
        for m_results in executor.map(single_remove_eyeblinks, eeg_list, idx_mat_list, n_clusters_list, fd_threshold_list, ssa_threshold_list, svd_list, antidiag_list):
            artifact[i,:] = m_results
            i += 1
            # multithread_results = executor.map(single_remove_eyeblinks, eeg_list, idx_mat_list, n_clusters_list, fd_threshold_list, ssa_threshold_list, svd_list, antidiag_list) 
            # executor_done = cf.Future.done()
            # print(f'Done = {executor_done}')

    return artifact