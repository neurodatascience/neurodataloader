import nibabel as nb
from bids import BIDSLayout
import torch as T
import torch.utils.data.dataset
import pandas as pd
from string import Formatter
import os

class TorchBIDS(torch.utils.data.dataset.Dataset):
    # TODO: Need to decide on strategy for dealing/fetching with data from the tabular data
    def __init__(self, root_dir: str,
                 tabular_data_file: str=None, tabular_data_columns: list=None,
                 augmentation_list: list=None, bids_kwargs: dict=None,
                 get_kwargs: dict=None, match_entities: list=None, force_num_im: int=0):
        '''
        Pytorch Dataset for handling dataset in BIDS.
        Parameters
        ----------
        root_dir : str
            String indicating the root of the BIDS directory. Takes precedence over 'root' specified in
            'bids_kwargs'
        tabular_data_file : str
            Formatting string with the name of the subject-specific tabular files. The file should be in the subject
            directory: root_dir/sub-123/tabular_data.tsv. Format entries can be any BIDS entry.
            E.g.: tabular_data_file='sub-{subject}_ses-{session}_tabular_penguin.tsv'
        tabular_data_columns : list
            List of columns to load from tabular data file.
        augmentation_list : list
            Currently unused; to be implemented in next release. List of functions to apply to data before being
             returned
        bids_kwargs : dict
            Optional. Keyword arguments to be passed to the bids.BIDSLayout object initialization.
        get_kwargs : dict
            Optional. Keyword arguments to be passed to the 'get' method of bids.BIDSLayout. It is used for specifying
            modality, session, task, etc. to be taken from the full dataset.
        match_entities : list
            Optional. When accessing the list of data, images will be grouped together when the entities specified in
            'match_entities'. For example, if ['subject'] is specified, all images from the subject will be grouped
            together. ['subject','session','suffix'] would group together images from the same subject, session, and
            suffix (modality).
        force_num_im : int
            Optional. Forces the image set to be of a certain size; rejects sets that aren't of that size. This is
            useful in cases where there is missing data (e.g., "return only images which have both AP and PA
            acquisitions")
        '''

        if(bids_kwargs is None):
            bids_kwargs = {}
        if(get_kwargs is None):
            get_kwargs = {}
        bids_kwargs['root'] = root_dir
        # Glorified Pybids wrapper
        self.bidsdata = BIDSLayout(**bids_kwargs)
        self.tabular_data_file = tabular_data_file
        self.tabular_data_columns = tabular_data_columns
        self.get_kwargs = get_kwargs
        self.augmentation_list = augmentation_list
        if('extension' not in self.get_kwargs.keys()):
            self.get_kwargs['extension'] = ['nii.gz', 'nii']
        file_list = self.bidsdata.get(**self.get_kwargs)
        self.image_set = []
        if(match_entities is None):
            self.image_set = file_list
        else:
            # Iterate through valid file list; get list of matches
            acc_ind_list = []
            for ind, f in enumerate(file_list):
                if(ind in acc_ind_list):
                    continue
                else:
                    acc_ind_list.append(ind)
                    flist = [f]
                match_dict = {}
                absent_set = set()  # for entities that we want to make sure match but aren't in all images (e.g. acq)
                for match_ent in match_entities:
                    if(match_ent not in f.entities.keys()):
                        absent_set.add(match_ent)
                        continue
                    match_dict[match_ent] = f.entities[match_ent]
                if('extension' not in match_entities):  # get images only
                    match_dict['extension'] = ['nii.gz', 'nii']
                match_list = self.bidsdata.get(**match_dict)
                for m in match_list:
                    if(m in file_list and m not in flist):
                        # check if it has any entities in the absent list; skip if so
                        if(len(set(m.entities.keys()).intersection(absent_set)) > 0):
                            continue
                        flist.append(m)
                        acc_ind_list.append(file_list.index(m))
                if(force_num_im > 0 and len(flist) != force_num_im):  # if force_num_im is specified, check for match
                    continue
                self.image_set.append(flist)

        # Tabular
        if(self.tabular_data_file is not None):
            self.tabular_keys = [ent[1] for ent in Formatter().parse(self.tabular_data_file) if ent[1] is not None]
        else:
            self.tabular_keys = None
        return

    def __len__(self):
        return len(self.image_set)

    def __getitem__(self, idx):
        '''
        Loads full entry in self.image_set; returns images as Pytorch tensors and meta data as dicts
        Parameters
        ----------
        idx : int
            Index of item to load.
        Returns
        -------
        ret_im : list
            List of image tensors.
        ret_meta : list
            List of meta data associated with the images in ret_im
        ret_tabular : list
            List of Pandas dataframes, each containing single-subject tabular data.
        '''
        # Load images; convert to Pytorch
        ret_im = []
        ret_meta = []
        ret_tabular = []
        for im in self.image_set[idx]:
            # Get image & image metadata
            ret_im.append(T.Tensor(nb.load(im).get_fdata()))
            ret_meta.append(im.get_metadata())

            # Load tabular data
            if (self.tabular_keys is None):
                continue
            tabdict = {}
            iment = im.get_entities()
            for k in self.tabular_keys:
                tabdict[k] = iment[k]
            # Form expected filename
            tabfname = self.tabular_data_file.format(**tabdict)
            tabind = im.dirname.rfind('sub-{subject}'.format(subject=iment['subject']))
            tabdir = im.dirname[:tabind] + 'sub-{subject}'.format(subject=iment['subject']) + os.sep
            tabfile = tabdir + tabfname
            ret_tabular.append(pd.read_csv(tabfile, sep='\t', usecols=self.tabular_data_columns))
        return ret_im, ret_meta, ret_tabular
