[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DNAS-bioDataSci/TREDNet_v1.1-colab/blob/main/TREDNet.ipynb)

- 🧬 GWAS → fine-map🔻🧣→ **TREDNet** 🤖💎 → assays 👩‍🔬🧫 → translation 💊

TREDNet is a genomics tool developed at NIH for functional studies with translational potential. It is particularly powerful for diseases where noncoding enhancer variants drive pathology, like type 2 diabetes (T2D).  This repo contains BioDataSci's port of the original TREDNet to Google Colab's NVIDIA-GPU environment.

# TREDNet_v1.1-colab Setup Instructions

Launch `TREDNet.ipynb` notebook by clicking [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DNAS-bioDataSci/TREDNet_v1.1-colab/blob/main/TREDNet.ipynb). It builds and runs a TREDNet deep learning pipeline on your own Colab managed GPU instance.

The Colab jupyter notebook:
- installs from github TREDNet python code, this file, and large model files
- builds the required TREDNet Conda/Tensorflow environment on a target GPU host 
- pulls hg38 (human genome) reference data from a remote server into the GPU instance
- runs the deep learning pipeline on genomic input data and outputs classification results
- shows summary of the TREDNet classification performance results

# BioDataSci.com - AI/ML/Ops support

<table>
  <tr>
    <td align="center">
      <a href="https://www.biodatasci.com">
        <img src="https://static.wixstatic.com/media/15c2ba_d194f7241b7940a9ba65dee7d288ab61~mv2.png" width="50%" >
      </a>
    </td>
    <td>
      <h2>Your co-innovation partner in Biotech AI/ML Architecture and Operations</h2>
      <h3>https://bioDataSci.com  |  email: jeremy@bioDataSci.com</h3>
    </td>
  </tr>
</table>
