import nibabel as nb
import pandas as pd
import os
import torch as T
from collections import defaultdict as dd
from torch.utils.data.dataset import Dataset

class QuickBIDS(Dataset):
    def __init__(self, root_dir: str = None,
                 file_of_files: str = None,
                 entities_to_match: dict = None,
                 tabular_to_fetch: list = None,
                 verbose: bool = True,
                 device: str = 'cuda:0'):
        '''
        Creates a Pytorch-compatible dataset for a fixed BIDS directory.
        Parameters
        ----------
        root_dir : str
            Root directory of the BIDS data. Either root_dir or file_of_files must be defined.
        file_of_files : str
            Optional. File containing a list of the files to load. Avoids having to walk the directory.
        entities_to_match : dict
            Optional. If defined, will only select files which match the specified entities.
            Example: entities_to_match = {'sub':'1234'} will return only files with subject 1234.
        tabular_to_fetch : list
            Optional. List of str corresponding to column entries to fetch.
        get_metadata : bool
            Whether to fetch metadata from .json file associated with the image file.
        verbose : bool
            Whether to print dataset info.
        device : str
            Device to use for the Pytorch tensor.
        '''

        # We are making the following assumptions:
        # - Session is mandatory
        # - Assumes that tabular data is under sub-X/ directory (one below root, in the subject dir)
        #   - Also assumes that only one .csv is present.

        self.device = device
        self.tabular_to_fetch = tabular_to_fetch

        if root_dir is None and file_of_files is None:
            raise ValueError('Either root_dir or file_of_files must be defined.')

        self.file_list = []
        self.file_path_dict = dd(str)
        if(tabular_to_fetch is not None):
            self.tabular_path_dict = dd(str)
        else:
            self.tabular_path_dict = None

        if(file_of_files is None):
            for dirpath, _, files in os.walk(root_dir):
                for f in files:
                    ent_dict = self._entity_splitter(f)
                    # Select only images
                    if(f.endswith('.nii.gz') or f.endswith('.nii')):
                        if (entities_to_match is None):
                            self.file_list.append(f)
                            self.file_path_dict[f] = os.path.join(dirpath, f)
                        else:
                            for ent_name, ent_value in entities_to_match.items():
                                if(ent_dict[ent_name] == ent_value):
                                    self.file_list.append(f)
                                    self.file_path_dict[f] = os.path.join(dirpath, f)
                        if(tabular_to_fetch is not None):
                            # get directory
                            sub_name = ent_dict['sub']
                            sub_dir = dirpath.split('sub-' + sub_name)[0]
                            sub_dir = os.path.join(sub_dir, f'sub-{sub_name}')
                            print(sub_dir)
                            for s in os.listdir(sub_dir):
                                if(s.endswith('.csv')):
                                    self.tabular_path_dict[f] = os.path.join(sub_dir, s)
        else:
            # load from file
            f = open(file_of_files, 'r')
            files = f.read().splitlines()
            f.close()
            for fil in files:
                if(fil.endswith('.nii.gz') or fil.endswith('nii')):
                    split = fil.split(os.sep)
                    self.file_list.append(split[-1])
                    self.file_path_dict[split[-1]] = fil
                    if(tabular_to_fetch is not None):
                        ent_dict = self._entity_splitter(fil)
                        sub_name = ent_dict['sub']
                        sub_dir = fil.split(sub_name)[0]
                        for s in os.listdir(sub_dir):
                            if (s.endswith('.csv')):
                                self.tabular_path_dict[f] = os.path.join(sub_dir, s)


        if verbose: print(f'Found {len(self.file_list)} files')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        '''
        Loads image data in file_list / file_path_dict
        Parameters
        ----------
        idx : int
            Index of entry in file_list to load

        Returns
        -------
        torch.Tensor
            Imaging data placed on device specified at initialization
        dict
            Dictionary containing tabular
        '''
        file = self.file_list[idx]
        file_path = self.file_path_dict[file]
        tabular_path = self.tabular_path_dict[file]
        dat = T.Tensor(nb.load(file_path).get_fdata()).to(self.device)
        # get tabular data
        tab_dat = pd.read_csv(tabular_path, usecols=self.tabular_to_fetch).to_dict(orient='records')
        return dat, tab_dat


    @staticmethod
    def _entity_splitter(filename: str,
                         entity_delim: str = '_',
                         keyval_delim: str = '-') -> dict:
        '''
        Splits the input BIDS-compliant filename into its separate entities.
        Parameters
        ----------
        filename : str
            Filename to split
        entity_delim : str
            Delimiter between entities.  (in "sub-1234_ses-2", the underscore "_" is the entity delim)
        keyval_delim : str
            Delimiter between key-value pairs. (in "sub-1234_ses-2", the dash "-" is the keyval delim)
        Returns
        -------
        dict
            Dictionary with the found entities, and 'suffix' for the last entitiy (if unkeyed)
        '''
        spl = filename.split(entity_delim)
        entity_dict = dd(str)
        for ind, s in enumerate(spl):
            if(keyval_delim in s):
                k, v = s.split(keyval_delim)
                entity_dict[k] = v
            elif(ind == len(spl)-1):
                entity_dict['suffix'] = s.split('.')[0]
                entity_dict['extension'] = '.'.join(s.split('.')[1:])
        return entity_dict