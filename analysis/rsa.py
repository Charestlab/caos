import pandas
from analysis.order import en_order
from rsatoolbox.data.dataset import load_dataset
from rsatoolbox.rdm.calc import calc_rdm
from rsatoolbox.vis import show_rdm
from rsatoolbox.rdm.transform import rank_transform
import matplotlib.pyplot as plt

sub = 2
run = 1
atlas_name = 'HOCPA' ## Harvard-Oxford Cortical Probabilistic Atlas
fpath_labels = f'atlas_labels/{atlas_name}_labels.tsv'

## Load images and 
atlas_labels = pandas.read_csv(fpath_labels, sep='\t')
atlas_labels['map_val'] = atlas_labels.index + 1


dataset = load_dataset(f'data/ds_sub-{sub}_run-{run}.h5')
datasets = dataset.split_channel('regions')
rdms = calc_rdm(datasets, descriptor='conds')

## reorder
conds = rdms.pattern_descriptors['conds']
rdms.reorder([conds.index(o) for o in en_order if o in conds])
#rdms = rank_transform(rdms)

## plot a region's RDM
plt.close('all')
rdms1 = rdms.subset('regions', [45, 46]) ## IT = [14, 15, 16]
fig, axes, handles = show_rdm(rdms1, rdm_descriptor='regions', pattern_descriptor='conds', figsize=(12,12), show_colorbar='panel')
plt.savefig(f'plots/rdms-sub-{sub}_run-{run}.png', dpi=200)




