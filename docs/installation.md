# Download Data
**DWD** are shipped in two formats: `.parquet` and `.zarr`. Both has their own advantages and drawbacks:

* `.parquet` files are fully supported by Hugging Face, and can be downloaded in subfiles by parameters, but are unoptimized in our data interface.
* `.zarr` files enable fast querying and processing, but they must be downloaded as a complete archive representing an entire network.

To download, please visit the Hugging Face [repository](https://huggingface.co/datasets/rugds/ditec-wdn/tree/main).

# Instal the Data Interface
DiTEC-WDN is available for Python >= 3.10. Please setup a virtual environment before installation.

As some libraries are tailored to your OS and CUDA, user should install them separately as follows:

1. Install [PyTorch >= 2.3](https://pytorch.org/get-started/locally/)
2. Instal [PyG >= 2.3](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

At this time, DiTEC-WDN works best on PyTorch and PyG 2.3.

Afterwards, you can clone DiTEC-WDN or install it via pip:

```python
pip install git+https://github.com/DiTEC-project/DiTEC_WDN_dataset.git
```

Then, required libraries are listed in `requirements.txt` that can be downloaded using this command:

```python
pip install -R requirments.txt
```

Tada! DiTEC-WDN data interface has been installed!

