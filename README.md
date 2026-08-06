## Shadow models for quantum neural networks for cloud cover parametrization

This repository contains the code for shadow models for quantum neural networks for cloud cover parameterization. Shadow models are classical models, that reproduce the input-output relations of the QML model, to allow for later coupling to the earth system model to circumvent limited quantum hardware availability. 

It is part of the paper "Shadow models of a quantum model for cloud cover and the influence of finite sampling noise", submitted to New Journal of Physics.

The code is based on the QML-based cloud cover-parameterization from Pastori et al. 2026 Mach. Learn.: Earth 2 015008 [10.1088/3049-4753/ae4981](https://doi.org/10.1088/3049-4753/ae4981), with corresponding code repository [10.5281/zenodo.21504569](https://doi.org/10.5281/zenodo.21504569). Used here, are the QNN's, functions for pre- and post-processing, as well as for data loading.



# Usage

Here is the list of commands you should run in order to create the correct Python environment to run these codes.


`conda create --name mycondaenv python=3.11.0`

`conda activate mycondaenv`

`pip install -r requirements.txt`

`python -m ipykernel install --user --name my-kernel --display-name="My Kernel"`

`conda deactivate`

Now you should be able to see the kernel ‘my-kernel’ in your list of Jupyter kernels, and you can select that to run your notebooks.

# Code Overview

The directory is structured as follows:
- The folder '\clc_functions' contains functions from [10.5281/zenodo.21504569](https://doi.org/10.5281/zenodo.21504569) in order to use the QNN for which we develop the shadow models. It includes functions for data loading, input and output transforms, as well as the layouts for the QNN's.
- The folder '\data_hardware\output' contains the output computed by quantum hardware (Euro-Q-Exa). In particular, it includes:
    - the evaluations of one QNN on a smaller test data set of 4000 points repeated over several days 'results_clc_i_varreg_date_processed' numbered by the date and different cloud regimes 'i'. Please note, that this includes the expectations value, the values after the affine linear layer as well as the final post-processed predictions, but not the counts from the measurements. Repeated experiments are denoted with 'results_clc_i_varreg_date_2_processed'
    - the evaluation of QNN on the grid to compute the shadow model using the quantum fourier model (QFM) 'Yft_flat_...' numbered by the date.
- The subfolder \data_hardware\shadow contains the data for the corresponding shadow models (for the Pytorch model). It can be evaluated with the notebook 'qfm_pytorch_model_eval.ipynb'.
    
    
- The folder '\optimal_params' contains the parameters of the QNN's with ansatzes XYZ and ZZXY for 6 features trained in a noiseless setting and with 1000 shots and variance regularization provided by L.Pastori [10.1088/3049-4753/ae4981](https://doi.org/10.1088/3049-4753/ae4981). 
- To run the code, a folder '\test_data' should be created in which the data cloud_regimes from the repository [10.5281/zenodo.21455691](https://doi.org/10.5281/zenodo.21455691) are saved. This is a subset of the test data used in [10.5281/zenodo.21504569](https://doi.org/10.5281/zenodo.21504569) but split into four different cloud regimes.

The following files are included:
- the file 'cs_functions_jit.py' containes functions to compute the shadow models via the QFM using the fast Fourier transform (FFT).
- the files 'quasi_interpolation.py' containes functions to comput the shadow models via quasi interpolation on a full grid or a sparse grid.

Lastly, we include the following notebooks:
- To evaluate the QNN on the test data, for reference values to compare against the shadow models:
    - 'ref_qml6f.ipynb' to evaluate all 4 cloud regimes, the file 'ref_qml_6fN1000'.ipynb, evaluates only the first 1000 points of each cloud regime.
- Shadowing with a piecewise affine-linear quasi-interpolation:
    - 'intp_pw_linear.ipynb'
- Shadowing with a quasi-interpolation on sparse grids: 
    - 'intp_sparse_data.ipynb' to generate and save the sparse grid and evaluate the QNN on the sparse grid
    - 'intp_sparse_shadow.ipynb' to evaluate the shadow model on the test set
- Shadowing with the QFM using FFT:
    - 'qfm_trunc_shadow.ipynb' to compute and evaluate the shadow model using QFM and FFT
    - 'qfm_for_pytorch.ipynb' computes coefficients for a shadow model and saves it to be used in a pytorch model
    - 'qfm_pytorch_model_eval.ipynb' defines the Fourier-based shadow model as a pytorch model (for a faster evaluation of the test set)
    - 'qfm_hardware_plots.ipynb' a function to plot the outputs of the QML models from Euro-Q-Exa


## Data
For reproducibility, the training and test data used by Pastori et al. is published in the accompaning data repository [10.5281/zenodo.21455691](https://doi.org/10.5281/zenodo.21455691) supplementing the repository [10.5281/zenodo.21504569](https://doi.org/10.5281/zenodo.21504569). It is based on the DYAMOND data set [10.1186/s40645-019-0304-z](https://doi.org/10.1186/s40645-019-0304-z), coarse-grained to R2B5 following Grundner et al. [10.1029/2023MS003763](https://doi.org/10.1029/2023MS003763).



The output generated on quantum hardware (Euro-Q-Exa) is also included in the subfolder '\data_hardware' for reproducibility of the results.

The optimal parameters computed in the training regimes in Pastori et al. 2026 Mach. Learn.: Earth 2 015008 DOI [10.1088/3049-4753/ae4981](https://doi.org/10.1088/3049-4753/ae4981) are included in '\optimal_params'


## Authors and acknowledgment

Authors: 
- Hedwig Keller for code regarding the shadow models
- Lorenzo Pastori for code regaring the QNN's, post- and pre-processing as well as data loading.
    The files from the folder '\clc_functions' are authored by Lorenzo Pastori from the repository [10.5281/zenodo.21504569](https://doi.org/10.5281/zenodo.21504569), but included here to ensure that the repository is self-contained. (Please note, that in the extraction of the test data set into different cloud regimes no random seed was set, so a exact replication of the cloud regime data set is not possible.)

The corresponding training and test data set can be found at [10.5281/zenodo.21455691](https://doi.org/10.5281/zenodo.21455691)
 It is based on the DYAMOND data set [10.1186/s40645-019-0304-z](https://doi.org/10.1186/s40645-019-0304-z), coarse-grained to R2B5 following Grundner et al. [10.1029/2023MS003763](https://doi.org/10.1029/2023MS003763).




## License

Apache-2.0 license

