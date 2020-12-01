# neurodataloader
neurodataloader is a combination of [Pytorch](https://github.com/pytorch/pytorch) datasets and [Pybids](https://github.com/bids-standard/pybids) intended to facillitate loading BIDS-compliant datasets for images.  
  
The Dataloader supplies sets of images grouped by the user's requirements; for example, sets are grouped by subject, session, and/or imaging modality. For datasets with exceptionally-large tabular data, users can use subject-specific .tsv files to allow real-time loading.