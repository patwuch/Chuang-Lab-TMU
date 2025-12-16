# Chuang Lab @ Taipei Medical University, Department of Molecular Parasitology and Tropical Diseases

We focus on modelling the relationship between dengue fever epidemiology and climate change by analyzing geospatial data, infectious reports text data, government recorded infection cases, and Shared Socioeconomic Pathways models. We leverage both data analysis, machine learning, and deep learning methods to better capture how long term climate trends observed in the past might predict the spread of dengue fever spatiotemporally in the future.

The repository consists of four modules, each contributing to a part of the aforementioned ecosystem of data preprocessing, transformation, and eventual application in machine learning.

Each module has a 'main' side where code resides and 'work' side where raw data, transformed data, or data visualisations resides. [Snakemake](https://snakemake.readthedocs.io/en/stable/) is used for data provenance tracking.

The dataflow of a fully realized pipeline would combine outputs from module (A), (B), (C) as well as government-recorded dengue fever cases to create models with module (D). 

## Active Modules

(A) Historical Climate Data Module
* Zonal Statistics via Google Earth Engine
* TCCIP Historical Downscaling Climate Data

(B) Future Climate Predictive Data Module
* IAMC/IPCC Shared Socioeconomic Pathways Models
* TCCIP () Shared Socioeconomic Pathways Models

(C) Dengue Fever Infectious Reports Natural Language Processing Module

(D) Dengue Fever Machine Learning Module


## Legacy Projects / Collaborations

* Impact of Weather Condition on Vehicular Accidents in Taiwan
