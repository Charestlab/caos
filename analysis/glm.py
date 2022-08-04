""""

# The atlas namings correspond to the original FSL’s acronyms for them (
# HOCPA=Harvard-Oxford Cortical Probabilistic Atlas; 
# HOCPAL=Harvard-Oxford Cortical Probabilistic Atlas Lateralized; 
# HOSPA=Harvard-Oxford Subcortical Probabilistic Atlas
# )
"""

from os.path import join, expanduser
import nibabel, nilearn.image, pandas, numpy
from scipy.interpolate import pchip
from templateflow import api as tflow_api
from rsatoolbox.data.dataset import Dataset


sub = 2
run = 1
tr = 1.2
space = 'MNI152NLin2009cAsym'
atlas_name = 'HOCPA'
bids_dir = expanduser('/Volumes/T7/caos/BIDS')
prep_dir = join(bids_dir, 'derivatives', 'fmriprep')
fname_evts = f'sub-{sub}_task-exp_run-0{run}_events.tsv'
fname_bold = f'sub-{sub}_task-exp_run-0{run}_bold_space-{space}_preproc.nii.gz'
fname_mask = f'sub-{sub}_task-exp_run-0{run}_bold_space-{space}_brainmask.nii.gz'
fpath_bold = join(prep_dir, f'sub-{sub}', 'func', fname_bold)
fpath_mask = join(prep_dir, f'sub-{sub}', 'func', fname_mask)
fpath_evts = join(bids_dir, f'sub-{sub}', 'func', fname_evts)
fpath_atlas = tflow_api.get(space, resolution=2, atlas=atlas_name, desc='th25')
fpath_labels = f'atlas_labels/{atlas_name}_labels.tsv'

## Load images and 
atlas_labels = pandas.read_csv(fpath_labels, sep='\t')
atlas_labels['map_val'] = atlas_labels.index + 1
bold_img = nibabel.load(fpath_bold)
atlas_img = nibabel.load(fpath_atlas)
mask_img = nibabel.load(fpath_mask)
print('resampling bold..')
bold_resampled_img = nilearn.image.resample_img(
    bold_img,
    target_affine=atlas_img.affine,
    target_shape=atlas_img.shape,
    interpolation='continuous'
)
print('resampling mask..')
mask_resampled_img = nilearn.image.resample_img(
    mask_img,
    target_affine=atlas_img.affine,
    target_shape=atlas_img.shape,
    interpolation='nearest'
)

## get data out
bold3d = bold_resampled_img.get_fdata().astype(numpy.float32)
mask3d = mask_resampled_img.get_fdata() > 0.5
atlas3d = atlas_img.get_fdata().astype(int)

## reshape data to (volumes x voxels)
n_vols = bold3d.shape[-1]
xyz = bold3d.shape[:3]
data = bold3d[mask3d, :].T
atlas = atlas3d[mask3d]

## make design matrix
events_df = pandas.read_csv(fpath_evts, sep='\t')
events_df['trial_type'] = events_df.entity
block_dur = numpy.median(events_df.duration)

## convolve a standard HRF to the block shape in the design
STANDARD_HRF = numpy.load('hrf.npy')
STANDARD_TR = 0.1
hrf = numpy.convolve(STANDARD_HRF, numpy.ones(int(block_dur/STANDARD_TR)))

## timepoints in block (32x)
timepts_block = numpy.arange(0, int((hrf.size-1)*STANDARD_TR), tr)

# resample to desired TR
hrf = pchip(numpy.arange(hrf.size)*STANDARD_TR, hrf)(timepts_block)
hrf = hrf / hrf.max()

## make design matrix
conditions = events_df.trial_type.unique()
dm = numpy.zeros((n_vols, conditions.size))
all_times = numpy.linspace(0, tr*(n_vols-1), n_vols)
hrf_times = numpy.linspace(0, tr*(len(hrf)-1), len(hrf))
for c, condition in enumerate(conditions):
    onsets = events_df[events_df.trial_type == condition].onset.values
    yvals = numpy.zeros((n_vols))
    # loop over blocks
    for o in onsets:
        # interpolate to find values at the data sampling time points
        f = pchip(o + hrf_times, hrf, extrapolate=False)(all_times)
        yvals = yvals + numpy.nan_to_num(f)
    dm[:, c] = yvals

## add polynomials
#pdata = wdata / wdata.mean(axis=0)

## least square fitting
# The matrix addition is equivalent to concatenating the list of data and the list of
# design and fit it all at once. However, this is more memory efficient.
design = [dm]
data = [data.T]
X = numpy.vstack(design)
X = numpy.linalg.inv(X.T @ X) @ X.T

betas = 0
start_col = 0
for run in range(len(data)):
    n_vols = data[run].shape[0]
    these_cols = numpy.arange(n_vols) + start_col
    betas += X[:, these_cols] @ data[run]
    start_col += data[run].shape[0]



## export RSAtoolbox dataset
conds = events_df.entity.to_list()
dataset = Dataset(
    measurements=betas,
    obs_descriptors=dict(conds=conds),
    channel_descriptors=dict(regions=atlas)
)
dataset.save(f'data/ds_sub-{sub}_run-{run}.h5', overwrite=True)




