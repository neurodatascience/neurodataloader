# neurodataloader
`neurodataloader` is a set of utility developed by A. Hutton in the [NeuroDataScience:ORIGAMI](https://neurodatascience.github.io/) laboratory. It combines [Pytorch](https://github.com/pytorch/pytorch) datasets and [Pybids](https://github.com/bids-standard/pybids) to facillitate loading BIDS-compliant datasets. It is particularly useful for the [UK Biobank](https://github.com/neurohub/neurohub_documentation/wiki/1.2.UKBiobank-Access-Request) [NeuroHub](https://neurohub.ca/) project. 

Instead of loading individual images, the Dataloader supplies sets of images grouped by the user's requirements (e.g. return images grouped by subject, by session, and/or imaging modality). This relies on the BIDS specifications. The UK Biobank data are available on [BIDS](https://bids.neuroimaging.io/) format on Compute Canada Beluga. See [NeuroHub documentation](https://github.com/neurohub/neurohub_documentation/wiki/1.2.UKBiobank-Access-Request).

In addition, individual tsv files for assessment and demographic information have been included in the BIDS hierarchy. The neurodataloader also implements extracting this information. 
