from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hazardEstimates",
    version="0.1",
    packages=find_packages(),
    #scripts=["Model.py", 'metrics.py', "raster_files.py", "Raster_Sets.py", "tif_datasets.py", "XY_Dataset.py"],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=["scikit-learn==0.22.2post1", "sklearn", "matplotlib", "numpy", "pandas", "rasterio","joblib"],


    # metadata to display on PyPI
    author="William Mobley",
    author_email="wmobley@tamu.edu",
    description="This package helps predict spatial flood hazard using rasters and Machine Learning ",
    keywords="hello world example examples",
    url="https://github.tamu.edu/wmobley/Hazard_Estimates",   # project home page, if any
    project_urls={
        "Bug Tracker": "https://github.com/wmobley/Hazard_Estimates/issues",
        "Documentation": "tbd",
        "Source Code": "https://github.com/wmobley/Hazard_Estimates",
    },
    classifiers=[
        "License :: OSI Approved :: Python Software Foundation License"
    ],
    python_requires = '>=3.6',

# could also include long_description, download_url, etc.
)