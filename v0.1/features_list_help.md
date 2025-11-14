# Features List 

## Overview

The `features_list.txt` file is a comprehensive catalog of genomic features used as input for the TREDNet (Transcription Factor Binding Prediction) model. This file contains metadata about various types of genomic data from the ENCODE project that are used to train or make predictions with the model.

## File Structure

The file is tab-separated and contains **4,562 lines** (including header) with the following 6 columns:

1. **IND** - Index number (ranging from 1 to 4,561)
2. **SRC_TYPE** - Feature source type (see below)
3. **TARGET** - Specific feature/target name
4. **CL_ID** - Cell line or tissue ontology identifier (EFO, CLO, CL, or UBERON format)
5. **FILE_ACCESSION** - ENCODE file accession number (ENCFF format)
6. **CL_NAME** - Human-readable cell line or tissue name

## Feature Types

The file contains three main categories of genomic features:

### 1. DHS (DNase I Hypersensitive Sites)
- Represents open chromatin regions
- Indicates areas of the genome that are accessible to transcription factors
- Examples: Various cell lines (A549, K562, HeLa-S3, HepG2, GM12878, etc.)

### 2. HM (Histone Modifications)
- Various histone marks that indicate chromatin state and gene regulation
- Common modifications include:
  - **H3K27ac** - Active enhancers
  - **H3K4me3** - Active promoters
  - **H3K4me1** - Enhancers
  - **H3K9me3** - Heterochromatin/repressed regions
  - **H3K27me3** - Polycomb-repressed regions
  - **H3K36me3** - Active gene bodies
  - **H2AFZ** - Histone variant
  - And many other modifications

### 3. TF (Transcription Factors)
- ChIP-seq data for specific transcription factors
- Examples include:
  - **CTCF** - Chromatin organizer
  - **POLR2A** - RNA Polymerase II
  - **ESR1** - Estrogen receptor
  - **FOXA1** - Forkhead box protein
  - **STAT1, STAT2, STAT3** - Signal transducers
  - **MYC** - Proto-oncogene
  - And many other transcription factors

## Data Sources

All entries in this file reference data from the **ENCODE (Encyclopedia of DNA Elements)** project:
- Each entry has a unique ENCODE file accession number (ENCFF format)
- Cell lines and tissues are identified using standard ontologies:
  - **EFO** - Experimental Factor Ontology
  - **CLO** - Cell Line Ontology
  - **CL** - Cell Ontology
  - **UBERON** - Uber-anatomy ontology

## Usage

This file serves as a reference catalog for:
- Extracting genomic signals from ENCODE datasets
- Training the TREDNet model
- Making predictions about transcription factor binding sites
- Mapping between feature indices and their corresponding data sources

## Example Entry

```
IND	SRC_TYPE	TARGET	CL_ID	FILE_ACCESSION	CL_NAME
1	DHS	-	EFO:0005713	ENCFF533SUL	8988T
```

This indicates:
- Index 1
- DNase I hypersensitive site data
- No specific target (indicated by "-")
- Cell line ID: EFO:0005713
- ENCODE file: ENCFF533SUL
- Cell line name: 8988T

