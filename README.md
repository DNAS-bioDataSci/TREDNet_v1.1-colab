# TREDNet_v1.1-colab Setup Instructions

See included Google Colab notebook `TREDNet.ipynb` to run TREDNet's deep learning pipeline on a Colab managed GPU instance.

The Colab jupyter notebook:
- installs from github TREDNet python code, this file, and large model files
- builds the required TREDNet Conda/Tensorflow environment on a target GPU host 
- pulls hg38 (human genome) reference data from a remote server into the GPU instance
- runs the deep learning pipeline against genomic input data and outputs classification results
- shows summary of the TREDNet classification performance results

Additionally included:
- system validation support incase of configuration errors

