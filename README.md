# TREDNet_v1.1-colab Setup Instructions

See included Google Colab notebook `TREDNet.ipynb` to run TREDNet's deep learning pipeline on a Colab managed GPU instance.

The Colab jupyter notebook:
- pulls supporting TREDNet code and large model files
- configures a specific Conda/Tensorflow GPU environment
- pulls hg38 (human genome) reference data from a remote server into the GPU instance
- runs the deep learning pipeline against genomic data and outputs classification results
- shows the classification performance results

Additionally:
- system validation support incase of configuration errors

