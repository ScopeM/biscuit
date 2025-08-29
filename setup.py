from setuptools import setup, find_packages
import platform
import os


def get_requirements(version = "core"):
    requirements = [
       # explicit, though TF will bring it in
        "tensorflow==2.18.0",
        "numpy>=1.26,<2",
        "scipy>=1.10,<1.12",
        "scikit-image>=0.22.0",
        "opencv-python>=4.9.0.80",
        "pandas>=2.1.0",
        "stardist>=0.9.1",
        "omnipose>=1.0.6",
        "tqdm>=4.66",
        "gitpython>=3.1.40",
        "coverage>=7.3",
        "mpl_interactions>=0.24",
        "ipympl>=0.9",
    ]

    if platform.processor() == "arm":
        requirements += ["tensorflow-metal"]
        
    # Check if installation is happening on euler
    if os.getenv("MIDAP_INSTALL_VERSION", "core").lower() == "euler":
        requirements = [
            "btrack==0.4.6",
            "coverage>=7.3.2",
            "gitpython>=3.1.40",
            "napari[all]",
            "omnipose==0.4.4",
            "opencv-python>=4.8.1",
            "pandas>=2.0.2",
            "scikit-image>=0.19.3,<=0.20.0",
            "stardist>=0.8.5",
            "tensorflow==2.15.0",
            "tqdm>=4.65.0",
            "build",
            "twine",
        ]

    return requirements


setup(
    name="biscuit",
    version="1.0.0",
    description="A package for segmentation comparison",
    long_description=""" """
    packages=find_packages(),  # Add this line
    install_requires=get_requirements(),  # Add this line
)
