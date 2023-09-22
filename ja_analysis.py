"""
JA analysis
============
Analyses a detector for Jitter and Accuracy.
At the heart is the jitter detection algorithm on annotation/detection
pairs. A Jitter of 0ms gives a 100% score and then drops to
zero with increased jitter.
Missed beats and extra/spurios detections are then used to calculate Accuracy
as a normalised measure independent on the number of beats.
The overall score is then: JA = Jitter/% * Accuracy/%.
"""
import numpy as np
import util
from scipy import stats

# Used to determine how many beats could have been at max heartrate.
# This is needed to calculate the true negatives (TN).
maxHR = 220

# The jitter which gives a 50% performance. That's a 1/4 of the average
# RMSSD which is 40ms.
norm_jitter = 10E-3 # sec

a = 10 # number of annotated beats to trim from start
b = -5 # number of annotated beats to trim from end

# keys for the ja dict:
key_jitter = "jitter" # temproal jitter in s
key_accuracy = "accuracy" # statistical accuracy
key_ja = "ja" # ja score
key_tp = "TP" # True positives
key_tn = "TN" # True negatives
key_fp = "FP" # False positives
key_fn = "FN" # False negatives
key_mae = 'MAE' # Mean absolute error

def nearest_diff(annotation, nearest_match):
    # Calculates the nearest difference between values in two arrays and saves
    # index and sample position of nearest
    
    len_annotation=len(annotation)

    # Array which will contain matches for all annotations no matter if the actual detection is missing
    used_indices=[] 
    for i in range(len_annotation): # scan through 'source' peaks 
        diff = nearest_match - annotation[i] # subtract ith source array value from ALL nearest match values
        index = np.abs(diff).argmin() # return the index of the smallest difference value
        used_indices.append((index,np.abs(nearest_match[index]-annotation[i]))) # save as tuple in used_indices

    # Array which will contain unique instances by always choosing the one which has the shortest time
    # difference.
    # 
    unique_diffs=[]
    index_used=[]
    for j in used_indices:
        if not (j[0] in index_used):
            uni = []
            for k in used_indices:
                if k[0] == j[0]:
                    uni.append(k)
            i = np.argmin(uni,0)[1]
            unique_diffs.append(uni[i][1])
            index_used.append(j[0])
        
    return unique_diffs


def score(jitter,accuracy):
    """
    Calculates the JA score by multiplying the normalised jitter
    with the accuracy which in turn is based on missing and extra beats.
    """
    jitter_score = 1 / ( 1 + (jitter / norm_jitter) ) # normalised jitter 0..1
    return accuracy * jitter_score


def evaluate(det_posn, anno_R, fs, nSamples, trim=True):
    """
    JA analysis of interval variation, missed beat and extra detection positions
    det_posn: the timestamps of the detector in sample positions
    anno_R: the ground truth in samples
    fs: sampling rate of the ECG file
    nSamples: number of samples in the ECG file
    returns:
    ja[key_jitter]   : jitter in s
    ja[key_tp]       : true positive beats
    ja[key_tn]       : true negative beats
    ja[key_fp]       : false positive beats
    ja[key_fn]       : false negative beats
    ja[key_accuracy] : accuracy
    ja[key_ja]      : JA Score
    """
    # Do we have enough detections?
    if len(det_posn)<=10:
        warning='WARNING: Less than ten detections'
        print(warning)
        ja = {}
        ja[key_jitter] = 0
        ja[key_tp] = 0
        ja[key_tn] = 0
        ja[key_fp] = 0
        ja[key_fn] = 0
        ja[key_accuracy] = 0
        ja[key_ja] = 0
        ja[key_mae] = 0
        
        return ja

    # Median delay of the detection against the annotations
    delay_correction = util.calcMedianDelay(det_posn, anno_R)

    # Correction for detector delay
    det_posn = np.array(det_posn)-int(delay_correction) 

    # Trims 1st and last detections
    if trim==True:
        det_posn, anno_R = util.trim_after_detection(det_posn, anno_R, a, b)

    # Do we have enough detections?
    if len(det_posn)<=10:
        warning='WARNING: Less than ten detections'
        print(warning)
        ja = {}
        ja[key_jitter] = 0
        ja[key_tp] = 0
        ja[key_tn] = 0
        ja[key_fp] = 0
        ja[key_fn] = 0
        ja[key_accuracy] = 0
        ja[key_ja] = 0
        ja[key_mae] = 0
        
        return ja

    # Number of annotated R peaks
    len_anno_R = len(anno_R)

    # Number of detected R peaks
    len_det_posn = len(det_posn)

    # return anno / detector pairs
    anno_det_pairs = nearest_diff(anno_R, det_posn) 
    
    differences_for_jitter=[]
        
    for d in anno_det_pairs: 

        difference = np.abs(d / fs)
        differences_for_jitter.append(difference)

    ja = {}
    ja[key_mae] = np.mean(differences_for_jitter)
    ja[key_jitter] = stats.median_abs_deviation(differences_for_jitter)
    fp = len_det_posn - len(differences_for_jitter) # all detections - true positive = false positive
    fn = len_anno_R - len(differences_for_jitter) # all detections
    tp = len(differences_for_jitter)
    maxBeats = int(nSamples / fs * maxHR / 60)
    tn = max(0,maxBeats - (tp + fn + fp)) # remaining samples
    ja[key_tp] = tp
    ja[key_tn] = tn
    ja[key_fp] = fp
    ja[key_fn] = fn
    if (tp + tn + fp + fn) > 0:
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        ja[key_accuracy] = accuracy
        ja[key_ja] = score(ja[key_jitter],accuracy)
    else:
        ja[key_accuracy] = False
        ja[key_ja] = False
    print(ja)
    return ja
