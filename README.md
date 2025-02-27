# WITOIL-for-iMagine

[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/WITOIL-for-iMagine/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/WITOIL-for-iMagine/job/main)

WITOIL for iMagine is an advanced application utilizing the DEEPaaS API to predict the
transport and transformation of actual or hypothetical oil spills in the ocean, powered 
by the **Medslik-II** model.

The Medslik-II oil spill model is a community-driven, freely available framework designed
to predict the movement and weathering of oil slicks using a Lagrangian representation.
It simulates the dispersion of oil spills based on real-world environmental conditions.

## Integrating the model with DEEPaaS
In this repository, we have integrated a DeepaaS API into the existing Medslik-II oil spill model. To launch it, first install the package then run [deepaas](https://github.com/ai4os/DEEPaaS):

> ![warning](https://img.shields.io/badge/Warning-red.svg) **Warning**: If you are using a virtual environment, make sure you are working with the last version of pip before installing the package. Use `pip install --upgrade pip` to upgrade pip.

```bash
git clone https://github.com/ai4os-hub/WITOIL-for-iMagine
cd WITOIL-for-iMagine
git submodule init
git submodule update --remote --merge
pip install -e ./path/to/submodule/dir
pip install -e .
deepaas-run --listen-ip 0.0.0.0
```
## How To Use 
To use WITOIL for iMagine, users must first register and obtain essential datasets from the 
  following sources:
  1. **Copernicus Marine Environment Monitoring Service (CMEMS)**:
    - Users must create an account to access oceanographic data required for simulations.
       Registration is available [here](https://data.marine.copernicus.eu/register?redirect=%2Fproducts).Once the account is created, 
       the user should store their username and password for future use.

  <img src="https://raw.githubusercontent.com/ai4os-hub/WITOIL-for-iMagine/main/data/tutorial_images/CMEMS.png" alt="CMEMS Image" width="400"/>

  2. **European Centre for Medium-Range Weather Forecasts (ECMWF)**:
     - Users must register and obtain a token to access ERA5 reanalysis data. Registration 
       can be completed [here](https://accounts.ecmwf.int/auth/realms/ecmwf/protocol/openid-connect/auth?client_id=cds&scope=openid%20email&response_type=code&redirect_uri=https%3A%2F%2Fcds.climate.copernicus.eu%2Fapi%2Fauth%2Fcallback%2Fkeycloak&state=LnmYV9xerVidknPojo3UgHrUPSxQzlbc6x8GMlNWQis&code_challenge=KvF-CRFr9d7MJM4TMUq3sQOvBQZYIie4bB6dLJsSbtQ&code_challenge_method=S256).
     - Users must retrieve their token from their profile [here](https://cds.climate.copernicus.eu/profile) 
       under the "Personal Access Token" section. The last step, which is somewhat hidden, is to accept 
       the terms of use for the dataset. 
  
 
  <img src="https://raw.githubusercontent.com/ai4os-hub/WITOIL-for-iMagine/main/data/tutorial_images/access_token_ERA5.png" alt="CMEMS Image" width="400"/>
     
  - The user needs to navigate to the [ERA5 single layer page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download),
       scroll to the bottom of the page, and accept the terms of usage when they appear.
  
    <img src="https://raw.githubusercontent.com/ai4os-hub/WITOIL-for-iMagine/main/data/tutorial_images/Terms_of_usage_ERA5.png" alt="CMEMS Image" width="400"/>
## RUNNING WITOIL FOR iMAGINE
  Once registrations are complete, users must provide their credentials, spill details, and 
  environmental parameters to run the model. After entering the necessary information, users 
  can execute the simulation by clicking "Run". The outputs include visualizations of oil 
  dispersion, temporal oil concentrations, and current/wind direction vectors.This 
  process facilitates informed decision-making for mitigating the impact of oil 
  spills.
  
  <img src="https://raw.githubusercontent.com/ai4os-hub/WITOIL-for-iMagine/main/data/tutorial_images/oil_concentration_algeria.gif" alt="CMEMS Image" width="400"/>
 
## Project structure

```
├── Jenkinsfile             <- Describes basic Jenkins CI/CD pipeline
├── Dockerfile              <- Steps to build a DEEPaaS API Docker image
├── LICENSE                 <- License file
├── README.md               <- The top-level README for developers using this project.
├── VERSION                 <- Version file indicating the version of the model
│
├── witoil_for_imagine
│   ├── README.md           <- Instructions on how to integrate your model with DEEPaaS.
│   ├── __init__.py         <- Makes <your-model-source> a Python module
│   ├── ...                 <- Other source code files
│   └── config.py           <- Module to define CONSTANTS used across the AI-model python package
│
├── api                     <- API subpackage for the integration with DEEP API
│   ├── __init__.py         <- Makes api a Python module, includes API interface methods
│   ├── config.py           <- API module for loading configuration from environment
│   ├── responses.py        <- API module with parsers for method responses
│   ├── schemas.py          <- API module with definition of method arguments
│   └── utils.py            <- API module with utility functions
│
├── data                    <- Data subpackage for the integration with DEEP API
│
├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
│
├── models                  <- Folder to store your models
│
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
│                              the creator's initials (if many user development),
│                              and a short `_` delimited description, e.g.
│                              `1.0-jqp-initial_data_exploration.ipynb`.
│
├── references              <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated graphics and figures to be used in reporting
│
├── requirements-dev.txt    <- Requirements file to install development tools
├── requirements-test.txt   <- Requirements file to install testing tools
├── requirements.txt        <- Requirements file to run the API and models
│
├── pyproject.toml          <- Makes project pip installable (pip install -e .)
│
├── tests                   <- Scripts to perform code testing
│   ├── configurations      <- Folder to store the configuration files for DEEPaaS server
│   ├── conftest.py         <- Pytest configuration file (Not to be modified in principle)
│   ├── data                <- Folder to store the data for testing
│   ├── models              <- Folder to store the models for testing
│   ├── test_deepaas.py     <- Test file for DEEPaaS API server requirements (Start, etc.)
│   ├── test_metadata       <- Tests folder for model metadata requirements
│   ├── test_predictions    <- Tests folder for model predictions requirements
│   └── test_training       <- Tests folder for model training requirements
│
└── tox.ini                 <- tox file with settings for running tox; see tox.testrun.org
```
 

## References
  1. De Dominicis M., N. Pinardi, G. Zodiatis, R. Lardner. [MEDSLIK-II, a Lagrangian marine
     surface oil spill model for short-term forecasting - Part 1: Theory](https://doi.org/10.5194/gmd-6-1851-2013). 
     Geosci. Model Dev., 6, 1851-1869, 2013.
  2. De Dominicis M., N. Pinardi, G. Zodiatis, R. Archetti [MEDSLIK-II, a Lagrangian marine 
     surface oil spill model for short-term forecasting - Part 2: Numerical simulations and 
     validations](https://doi.org/10.5194/gmd-6-1871-2013). Geosci. Model Dev., 6, 1851-1869, 2013.  




