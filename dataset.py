import nibabel as nb
from bids import BIDSLayout
import torch as T
import torch.utils.data.dataset

class TorchBIDS(torch.utils.data.dataset.Dataset):
    # TODO: Need to decide on strategy for dealing/fetching with data from the tabular data
    def __init__(self, root_dir: str,  tabular_data: str=None, augmentation_list: list=None, bids_kwargs: dict=None,
                 get_kwargs: dict=None, match_entities: list=None, force_num_im: int=0):
        '''
        Pytorch Dataset for handling dataset in BIDS.
        Parameters
        ----------
        root_dir : str
            String indicating the root of the BIDS directory. Takes precedence over 'root' specified in
            'bids_kwargs'
        tabular_data : str
            Currently unused. To be implemented in next release.
        transform_list : list
            Currently unused; to be implemented in next release. List of functions to apply to data before being returned
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
        tuple
            tuple[0] is a list of tensors matching the images in self.image_set[idx]
            tuple[1] is a list containing the meta data of each image. tuple[0][2] would return the image data;
            tuple[1][2] would return the metadata for that same image.
        '''
        # Returns ([image], metadata_json)
        # Returned images
        # Load images; convert to Pytorch
        ret_im = []
        ret_meta = []
        for im in self.image_set[idx]:
            ret_im.append(T.Tensor(nb.load(im).get_fdata()))
            ret_meta.append(im.get_metadata())
        return ret_im, ret_meta