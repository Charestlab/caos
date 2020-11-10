from os.path import join, expanduser
import nibabel, pandas, numpy
from analysis.glm import make_design, whiten_data, fit_glm

sub = 1
run = 1
tr = 1.2
space = 'MNI152NLin2009cAsym'
bids_dir = expanduser('~/Data/caos/BIDS')
prep_dir = join(bids_dir, 'derivatives', 'fmriprep')
fname_evts = f'sub-{sub}_task-exp_run-0{run}_events.tsv'
fname_bold = f'sub-{sub}_task-exp_run-0{run}_bold_space-{space}_preproc.nii.gz'
fpath_bold = join(prep_dir, f'sub-{sub}', 'func', fname_bold)
fpath_evts = join(bids_dir, f'sub-{sub}', 'func', fname_evts)

## Load fmri and reshape data to (volumes x voxels)
img_data = nibabel.load(fpath_bold).get_fdata().astype(numpy.float32)
n_vols = img_data.shape[-1]
data = numpy.moveaxis(img_data, -1, 0).reshape([n_vols, -1])

## Load events from file and convolve with HRF
events = pandas.read_csv(fpath_evts, sep='\t')
design = make_design(events, n_vols, tr)

## Regress out polynomials
wdata, wdesign = whiten_data(data, design)

## Fit the GLM
betas = fit_glm(wdata, wdesign)

# unit-scale the data to be able to compare betas (within run)

## normalize betas


