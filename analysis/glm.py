from os.path import join, expanduser
import nibabel, nilearn.image, pandas, numpy
from templateflow import api as tflow_api
from analysis.glm_wrapper import make_design, whiten_data, fit_glm
from rsatoolbox.data.dataset import Dataset

# The atlas namings correspond to the original FSL’s acronyms for them (
# HOCPA=Harvard-Oxford Cortical Probabilistic Atlas; 
# HOCPAL=Harvard-Oxford Cortical Probabilistic Atlas Lateralized; 
# HOSPA=Harvard-Oxford Subcortical Probabilistic Atlas
# )

sub = 2
run = 1
tr = 1.2
space = 'MNI152NLin2009cAsym'
atlas_name = 'HOCPA'
bids_dir = expanduser('~/Data/caos/BIDS')
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
    target_shape=atlas_img.shape
)
print('resampling mask..')
mask_resampled_img = nilearn.image.resample_img(
    mask_img,
    target_affine=atlas_img.affine,
    target_shape=atlas_img.shape
)

## get data out
bold3d = bold_resampled_img.get_fdata().astype(numpy.float32)
mask3d = mask_resampled_img.get_fdata() > 0.5
atlas3d = atlas_img.get_fdata().astype(int)

## reshape data to (volumes x voxels)
n_vols = bold3d.shape[-1]
xyz = bold3d.shape[:3]
data = bold3d[mask3d].T
atlas = atlas3d[mask3d]

## Load events from file and convolve with HRF
events = pandas.read_csv(fpath_evts, sep='\t')
design = make_design(events, n_vols, tr)

## Regress out polynomials
wdata, wdesign = whiten_data(data, design)
## makes them into matrix???

## Transform to Percent Signal Change
## https://www.brainvoyager.com/bv/doc/UsersGuide/StatisticalAnalysis/TimeCourseNormalization.html
pdata = wdata / wdata.mean(axis=0)
## mean is not exactly 1, typically 0.995 - 1.005. problem?

## Fit the GLM
betas = fit_glm(wdata, wdesign)
# In [3]: betas.std(axis=0).mean() ## over timepoints
# Out[3]: 12.219756516863749
# In [4]: betas.std(axis=1).mean() ## over voxels
# Out[4]: 9.616672491255056

## export RSAtoolbox dataset
conds = events.entity.to_list()
dataset = Dataset(
    measurements=betas,
    obs_descriptors=dict(conds=conds),
    channel_descriptors=dict(regions=atlas)
)
dataset.save(f'data/ds_sub-{sub}_run-{run}.h5', overwrite=True)




