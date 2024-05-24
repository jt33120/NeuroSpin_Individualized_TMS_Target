import nibabel as nib
import numpy as np
import os
import nilearn
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import time
from nibabel import imagestats
from nilearn import datasets, plotting, surface, image, masking
from nilearn.maskers import NiftiSpheresMasker, NiftiMasker
from nilearn.surface import vol_to_surf
from nilearn.datasets import get_data_dirs
from nilearn.plotting import plot_epi
from plotly import graph_objects as go
from nilearn import plotting, surface, datasets
from nilearn import surface
from nilearn import plotting
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from nilearn.image import mean_img, math_img
from nilearn.plotting import plot_prob_atlas
from matplotlib import patches, ticker
from nilearn import datasets, plotting
from nilearn.image import get_data, index_img, mean_img,threshold_img
from nilearn.regions import Parcellations
from nilearn.plotting import view_img,plot_stat_map,show
from scipy.sparse import coo_matrix
import math
from nilearn.regions import RegionExtractor
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from scipy.spatial.distance import cdist
import seaborn as sns


def extract(FC,niimg,frontal_mask_img,fwhm,verbose = False) :
    """
    Apply extractor on an FCmap. We decrease the threshold by 0.01 until we get a cluster of size > 1000
    
    Parameters :
    fwhm : smoothing to use on the image
    
    Output : 
    best_cluster :  Nifti image
    best_target :  np.array 
    """
    print('Smoothing : ', fwhm)
    FC_smoothed = image.smooth_img(FC, fwhm)
    thr = np.max(FC_smoothed.get_fdata())
    while True:
        try:
            th_image = threshold_img(FC_smoothed, thr, two_sided=False)
            extractor = RegionExtractor(maps_img=th_image,
                                      mask_img=frontal_mask_img, 
                                      min_region_size=800, # en mm^3, 1cm x 1cm x 1cm
                                      threshold=thr, # bc inverted, > thr
                                      thresholding_strategy='img_value', 
                                      extractor='connected_components', 
                                      smoothing_fwhm=None, 
                                      standardize=False, 
                                      standardize_confounds=False, 
                                      detrend=False, 
                                      low_pass=None, 
                                      high_pass=None, 
                                      t_r=None, 
                                      verbose=0)
            extractor.fit();
            break  # Exit the loop if fit() succeeds
        except TypeError:
            thr -= 0.001  # Decrease the threshold
            #thr = max(0,thr)
    """
    if thr == 0 :# be sure it is processed at the end
        th_image = threshold_img(FC_smoothed, thr, two_sided=False)
        extractor = RegionExtractor(maps_img=th_image,
                                  mask_img=frontal_mask_img, 
                                  min_region_size=100, # en mm^3, 1cm x 1cm x 1cm
                                  threshold=thr, # bc inverted, > thr
                                  thresholding_strategy='img_value', 
                                  extractor='connected_components', 
                                  smoothing_fwhm=None, 
                                  standardize=False, 
                                  standardize_confounds=False, 
                                  detrend=False, 
                                  low_pass=None, 
                                  high_pass=None, 
                                  t_r=None, 
                                  verbose=0)
        extractor.fit();
    """
        
    print(thr)
    regions_extracted_img = extractor.regions_img_
    regions_index = extractor.index_
    n_regions_extracted = regions_extracted_img.shape[-1]
    
    cluster = Best_Cluster(n_regions_extracted, FC.affine, regions_extracted_img)
    if verbose :
        #plotting.plot_prob_atlas(
        #    regions_extracted_img, view_type="auto", cut_coords = ([-41, 43, 27]), title='Regions')
        sns.histplot(cluster.flatten(), kde=True, color='b', stat='density')
        plt.show()
        cluster_img = nilearn.image.new_img_like(FC, cluster, affine=FC.affine, copy_header=False)
        #plot_real_to_cash(cluster_img,niimg,([-41, 43, 27]),'seismic','Cluster and cash target')

        print('Thr = ' + str(thr) )
        print('Cluster size : ', np.count_nonzero(cluster))
        print('> 0 : ',np.count_nonzero(cluster[cluster>0]),'Max : ', np.max(cluster) ) 
        print('< 0 : ',np.count_nonzero(cluster[cluster<0]),'Min : ', np.min(cluster) ) 
        
    return cluster

def Best_Cluster(n,affine,clusters):
    """
    Return the  cluster wth the highest mean correlation, and the associated target as the voxel closest to all other voxwels in the cluster (minimal neighbour distance)
    
    Output :
        cluster : np.array
        target : np.array 3-D vector [x,y,z] in MNI-space

    """
    clusters = clusters.get_fdata()
    
    # GET CLUSTER WITH HIGHEST MEDIAN-FC
    medians = [ np.median(clusters[:,:,:,j]) for j in range(n) ] 
    best_FC_index = np.argmax(medians) # argmax bc FC are inverted
    
    # GET CENTER OF BEST TARGET, MINIMIZING NEIGHBOURS DISTANCE
    cluster = clusters[:,:,:,best_FC_index]

    
    return cluster

def get_center(cluster,affine):
    """
    Return the  cluster center using the predefined definition. Here, we define the center as the voxel minimizing distance to its neighbours.
    
    Output :
        target : np.array 3-D vector [x,y,z] in MNI-space

    """
    indices = np.where(cluster > 0)
    if len(indices[0]) == 0 :
        print(' WARNING ! EMPTY CLUSTER ')
        return np.array([])
    else :
        indices = np.transpose(indices)

            # Select a random subset of points (adjust the sample size as needed)
        sample_size = min(250, int(0.8*indices.shape[0]))  # Limit the sample size to avoid excessive computation
        sample_indices = np.random.choice(indices.shape[0], size=sample_size, replace=False)
        sample_points = indices[sample_indices]

        pairwise_distances = np.linalg.norm(sample_points[:, None] - sample_points, axis=-1)
            # Minimize the mean distance to this neighbors for every voxel in sample_points
        closest_point_index = np.argmin(np.sum(pairwise_distances, axis=1))
        best_target = tuple(sample_points[closest_point_index])
        return np.array(nilearn.image.coord_transform(best_target[0],best_target[1],best_target[2], affine))


def consensus_fwhm(FCmap,niimg,frontal_mask_img,fwhm1,fwhm2,verb):
    """
    Apply consensus on an FCmap to get best cluster and best target, based on extract and on different threshold level.
    Strategy : Blurred Intersection, for each cluster, half of the voxel must be in the intersection of the two other cluster
    We define the target as the voxel minimizing its distance to neighbours.
        
    Parameters :
    - w1, w2,w3 : weightes for the consensus
    - thr1, thr2, thr3, threshold for each extract call. !! thr > 0 because FC-mp are inverted
    
    Output : 
    - best_consensus_target : np.array [x,y,z]
    - best_consensus_cluster : np.array
    """
    
    # CALL EXTRACT ON THREE DIFFERENT THRESHOLDS
    best_cluster_1 = extract(FCmap,niimg,frontal_mask_img,fwhm1,verbose = verb)
    best_cluster_2 = extract(FCmap,niimg,frontal_mask_img,fwhm2,verbose = verb)
    best_cluster_3 = extract(FCmap,niimg,frontal_mask_img,6,verbose = verb)

    # APPLY CONSENSUS STRATEGY
    consensus_cluster = best_cluster_1 * best_cluster_2 * best_cluster_3
    # GET TARGET FROM CONSENSUS CLUSTER
    target = []
    
    # PLOT IF VERBOSE
    if verb:
        print('Max consensus : ',np.max(consensus_cluster))
        print('Min consensus : ',np.min(consensus_cluster))
        print('Size consensus : ', np.count_nonzero(consensus_cluster))

    return consensus_cluster

def bagging_non_overlap(data,niimg,frontal_mask_img,brain_masker,SgACC_time_series,verb,n=5):
    """
    Bagging after sclicing the time series into n non-overlaping chunks. 
    
    Parameters:
        n : number of maps to produce

    Output:
        Bagging cluster (numpy array) : mean of the targets obtained when applying consensus on time chunks
    """
    T = np.shape(data)[0]
    chunk_size = T // n  # Size of each chunk
    Best_Consensus_Cluster = []
    
    # COMPUTE CONSENSUS ON EACH CHUNK
    for i in range(n) :
        chunk = data[i*chunk_size:(i+1)*chunk_size, :]
        sgACC = SgACC_time_series[i*chunk_size:(i+1)*chunk_size, :]
        flatten_mean_fc_map = (np.dot(chunk.T, sgACC) / chunk_size)
        fc_map = brain_masker.inverse_transform(-flatten_mean_fc_map.T)
        best_consensus_cluster = consensus_main(fc_map,niimg,frontal_mask_img)
        Best_Consensus_Cluster += [best_consensus_cluster]
        
    # GET THE INTERSECTION OF EACH CONSENSUS
    intersection_array = Best_Consensus_Cluster[0]
    for array in Best_Consensus_Cluster[1:]:
        intersection_array = intersection_array * array
    
    
    if verb:
        cluster_img = nilearn.image.new_img_like(FCmap, intersection_array, affine=FCmap.affine, copy_header=False)
        #plot_real_to_cash(cluster_img,niimg,target,'seismic','Bagging and its associated target, n = '+str(n)+' & % = '+str(perc))
        
    return intersection_array.astype(int)

def bagging_overlap(data,niimg,frontal_mask_img,brain_masker,SgACC_time_series,verb,noise,fwhm1,fwhm2,perc=0.5,n=10):
    """
    Bagging after sclicing the time series into n overlaping chunks. 
    
    Parameters:
        n : number of maps to produce

    Output:
        Bagging cluster (numpy array) : mean of the targets obtained when applying consensus on time chunks
    """
    T = np.shape(data)[0]
    chunk_size = int(T * perc)  # Size of each chunk
    starts = np.linspace(0, T-chunk_size, num=n, endpoint=True, retstep=False, dtype=None, axis=0)
    Best_Consensus_Cluster = []
    
    # COMPUTE CONSENSUS ON EACH CHUNK
    for start in starts :
        chunk = data[int(start):int(start)+chunk_size, :]
        sgACC = SgACC_time_series[int(start):int(start)+chunk_size, :]
        correlation_matrix = sgACC * chunk
        flatten_fc_map = np.median(correlation_matrix, axis=0).reshape(-1, 1)
        #flatten_fc_map = (np.dot(chunk.T, sgACC) / sgACC.shape[0])

        if noise :
            inverted_fc_map,_,_ = generate_noisy_fc_maps(flatten_fc_map,brain_masker, mu = 0, sigma1=0.1,sigma2=0.05)
        #best_consensus_cluster = consensus_fwhm(inverted_fc_map,niimg,frontal_mask_img,fwhm1,fwhm2,False)
        best_consensus_cluster = extract(inverted_fc_map,niimg,frontal_mask_img,6,verbose = False)
        Best_Consensus_Cluster += [best_consensus_cluster]
    # GET THE INTERSECTION OF EACH CONSENSUS
    intersection_array = Best_Consensus_Cluster[0]
    for array in Best_Consensus_Cluster[1:]:
        intersection_array = intersection_array * array
    
    if verb:
        print('Size bagging : ', np.count_nonzero(intersection_array))
        #cluster_img = nilearn.image.new_img_like(FCmap, intersection_array, affine=FCmap.affine, copy_header=False)

    return intersection_array



"""
TESTING FUNCTIONS
"""


def generate_noisy_fc_maps(X,brain_masker, mu = 0, sigma1=0.1,sigma2=0.05):
    """
    Add noise to each time series in a 2D array following a mixed-effects model.

    Parameters:
        X (numpy array): vector of correlation coeffs (n_time_points, n_voxels).
        sigma1 (float): Standard deviation for voxel-wise variability.
        sigma2 (float): Standard deviation for spatial variability.
        mu : fixed effect

    Output:
        numpy array: Array with noise added to each time series.
    """
    # Calculate the fixed effects component (X * beta)
    beta = np.abs(np.random.randn(1))
    print(beta)
    fixed_effects = (sigma1**2) * np.dot(X, beta)

    # Generate random noise for within-subject variability (epsilon)
    epsilon = (sigma2**2) * np.random.randn(len(X))

    # Combine fixed effects, mu, and random effects to get the signal vector y
    flatten_noise = mu + fixed_effects + epsilon
    flatten_noise = flatten_noise.reshape(-1,1)

    inverted_fc_map = brain_masker.inverse_transform(-flatten_noise.T)
    inverted_fc_map = image.smooth_img(inverted_fc_map, 4)
    real_fc_map = brain_masker.inverse_transform(flatten_noise.T) # not inverted values
    real_fc_map = image.smooth_img(real_fc_map, 4)

    return inverted_fc_map,real_fc_map, flatten_noise


def compute_log_likelihood(expected_cluster, noisy_clusters, sigma1=0.1, sigma2=0.001):
    """
    Compute the log likelihood between an expected array and noisy clusters.

    Parameters:
        expected_cluster (array-like): Expected array.
        noisy_clusters (list of array-like): List of noisy clusters.
        sigma1 (float): Standard deviation for voxel-wise variability.
        sigma2 (float): Standard deviation for spatial variability.

    Output:
        float: Log likelihood.
    """
    # Calculate the number of noisy clusters
    num_clusters = len(noisy_clusters)
    
    # Initialize log likelihood
    log_likelihood = 0.0
    
    # Calculate the log likelihood for each noisy cluster
    for cluster in noisy_clusters:
        # Calculate the difference between the cluster and the expected array
        diff = np.array(cluster) - np.array(expected_cluster)
        
        # Calculate the voxel-wise variance
        var_voxelwise = sigma1 ** 2
        
        # Calculate the spatial variance
        var_spatial = sigma2 ** 2
        
        # Calculate the total variance
        total_variance = var_voxelwise + var_spatial
        
        # Calculate the log likelihood contribution for this cluster
        log_likelihood -= np.sum(diff ** 2) / (2 * total_variance) + \
                          np.sum(np.log(np.sqrt(2 * np.pi * total_variance)))
    
    # Adjust the log likelihood for the number of clusters
    log_likelihood -= 0.5 * num_clusters * len(expected_cluster) * np.log(2 * np.pi * total_variance)
    
    return log_likelihood

def accuracy(expected_cluster, noisy_clusters):
    """Return the likelihood of the model per label & the Root Mean Squared Error (RMSE)

    Parameters:
    Center_Coords : List of the target coordinates found on each image
    real_coords : [x,y,z] array of the real best target
    Labels : all parcellations, including the real one

    Output :
    LL:  log-likelihood score of the data under parcellation
    BIC: BIC score of the data under parcellation
    DIST : list of distance between found_coords, real_coords
    """
    N = len(expected_cluster)

    LL, BIC = np.zeros(N), np.zeros(N)
    
    # Compute likelihood
    cvll = compute_log_likelihood(expected_cluster[0],noisy_clusters[1:],0.1,0.05)
                
    # Compute BIC
    n_data_points = np.size(noisy_clusters[0]) * N
    bic = -2 * cvll + 3 * np.log(n_data_points) # 3 = n_parameters
    
    return bic, cvll


def reproducibility_image(expected_cluster, noisy_clusters):
    """ Run mutliple pairwise supervised ratings to obtain an average
    rating

    Parameters:
    Modeled_Data : NiftiImage of  modeled FC-map, noisy or bootstrap
    parcellation_template = NiftiImage of the template

    Output:
    ARI : ars score between the real parcellation on the ones obtained with the modified ones
    AMI : ami score between the real parcellation on the ones obtained with the modified ones
    
    """    
    N = len(noisy_clusters)
    AMI_values = []
    ARI_values = []
    
    # Perform bootstrap resampling and compute parcellation
    for i in range(N):

        # Compute AMI and ARI
        ami = adjusted_mutual_info_score(expected_cluster, noisy_clusters[i])
        ari = adjusted_rand_score(expected_cluster, noisy_clusters[i])
        AMI_values.append(ami)
        ARI_values.append(ari)
    return AMI_values,ARI_values

    
"""
PLOTTING FUNCTIONS
"""

def plot_real_to_cash(sub_test,niimg,ctr,cMap,Title) :
    """ 
    Surface plot of the recommended target next to the real stim target. Here the real stim target is cash target ([-41,43,27]).
    rating
    """ 
    fsaverage = datasets.fetch_surf_fsaverage()
    curv_left = surface.load_surf_data(fsaverage.curv_left)
    curv_left_sign = np.sign(curv_left)
    texture = surface.vol_to_surf(sub_test, fsaverage.pial_left)

    pial = fsaverage.pial_left
    coord = np.array([-41, 43, 27])

    mesh_coords = np.array(nib.load(pial).darrays[0].data)

    diff = mesh_coords-coord
    sq_distance = np.sum(diff**2, 1)
    closest = np.argsort(sq_distance)[:3]
    parcellation = np.zeros(texture.shape)
    parcellation[closest] = 1
    diff = mesh_coords-ctr
    sq_distance = np.sum(diff**2, 1)
    closest = np.argsort(sq_distance)[:3]
    parcellation2 = np.zeros(texture.shape)
    parcellation2[closest] = 1

    fig = plotting.plot_surf_stat_map(surf_mesh = fsaverage.infl_left, stat_map=texture,hemi='left', view=[10.0, 120.0],title=Title,engine = 'matplotlib', cmap = cMap)
    fig_bis=plotting.plot_surf_contours(fsaverage.infl_left, parcellation, labels=['roi'], levels=[1], figure=fig, legend=False, colors=['g'])
    fig_finale=plotting.plot_surf_contours(fsaverage.infl_left, parcellation2, labels=['roi'], levels=[1], figure=fig_bis, legend=False, colors=['c'])
    fig_finale.show()
    return fig_finale

