from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hazardEstimates",
    version="0.1",
    packages=["Model", 'metrics', "raster_files", "Raster_Sets", "tif_datasets", "XY_Dataset"]
    scripts=["Model.py", 'metrics.py', "raster_files.py", "Raster_Sets.py", "tif_datasets.py", "XY_Dataset.py"],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=["sklearn>=0.0", "numpy==1.18.2", "pandas==1.0.3", "rasterio==1.1.3"],


    # metadata to display on PyPI
    author="William Mobley",
    author_email="wmobley@tamu.edu",
    description="This package helps predict spatial flood hazard using rasters and Machine Learning ",
    keywords="hello world example examples",
    url="https://github.tamu.edu/wmobley/Hazard_Estimates",   # project home page, if any
    project_urls={
        "Bug Tracker": "https://github.tamu.edu/wmobley/Hazard_Estimates/issues",
        "Documentation": "tbd",
        "Source Code": "https://github.tamu.edu/wmobley/Hazard_Estimates",
    },
    classifiers=[
        "License :: OSI Approved :: Python Software Foundation License"
    ],
    python_requires = '>=3.6',

# could also include long_description, download_url, etc.
)