# Biscuit Changelog

## [1.0.0] - 2024-12-19

### 🚀 **Project Initialization**
- **Created Biscuit project** - A streamlined, notebook-focused image segmentation tool
- **Repository cleanup and optimization** for focused functionality
- **Git/GitHub dependencies removal** for fresh repository initialization

### 🧹 **Repository Cleanup (Purge Process)**

#### **Removed Git/GitHub Dependencies:**
- Deleted `.github/` directory with CI/CD workflows
- Removed all `.gitignore` files from root and subdirectories
- Cleaned repository for fresh GitHub upload

#### **Removed Unused Directories:**
- `training/` - Training scripts and notebooks (not needed for core functionality)
- `euler/` - Euler cluster specific configurations
- `tests/` - Test suite (not needed for production)
- `img/` - GUI images and assets
- `midap/networks/` - Neural network architectures (not used by notebooks)
- `midap/tracking/` - Cell tracking modules (not used by notebooks)
- `midap/imcut/` - Image cutting utilities (not used by notebooks)
- `midap/correction/` - Segmentation correction tools (not used by notebooks)
- `midap/data/` - Data processing pipelines (not used by notebooks)

#### **Removed Unused Files:**
- `setup.py` - Package configuration (not needed for standalone notebooks)
- `environment.yml` - Conda environment file
- `midap/main*.py` - Main execution scripts
- `midap/checkpoint.py` - Checkpoint functionality
- `midap/config.py` - Configuration management
- Most files in `midap/apps/` (kept only `PySimpleGUI.py` and `segment_cells.py`)

### 📊 **Optimization Results:**
- **Size reduction:** From ~6.9MB to ~5.6MB (19% reduction)
- **Focused functionality:** Only essential modules for notebook execution
- **Clean structure:** Streamlined for GitHub Pages deployment

### 🎯 **Core Functionality Preserved:**
- **Jupyter Notebooks:** `biscuit_google_colab.ipynb` and `biscuit_eth_euler.ipynb`
- **Segmentation Modules:** All segmentation algorithms and tools
- **Data Examples:** Sample images for testing
- **Essential Dependencies:** Core functionality for image analysis

### 🔧 **Technical Details:**
- **Dependency Analysis:** Thorough analysis of import chains
- **Safe Removal:** Verified no critical dependencies were broken
- **Import Testing:** Confirmed package imports correctly
- **Notebook Compatibility:** Maintained full functionality of core notebooks

---

## **Project Genesis:**
**BISCUIT (BioImage Segmentation Comparison Utility and Interactive Tool)** traces its roots back to the **Scientific IT Services (SIS)** at ETH Zurich. Spearheaded by **Franziska Oschmann**, the goal was to create an intuitive yet powerful platform for visually comparing cell segmentation models on microscopy data.

The initial version of BISCUIT was built on top of the **Microbial Image Data Analysis Pipeline (MIDAP)** framework—a modular system originally developed for analyzing data from Mother Machine experiments. MIDAP's flexible architecture made it a solid foundation for building interactive segmentation workflows.

Development continued in close collaboration with the ETH Zurich imaging center **ScopeM**, with practical contributions and technical leadership by **Szymon Stoma** and **Andrzej Rzepiela**. Together, the team laid the groundwork for a tool that aspires to become a versatile visual benchmarking suite for segmentation models in the Life Sciences.

Rather than creating a traditional fork, we decided to create a clean, optimized copy that focuses specifically on the core segmentation functionality through Jupyter notebooks. This approach allows us to maintain a focused codebase while preserving the powerful segmentation algorithms that have proven effective in biological image analysis.

The project incorporates proven segmentation algorithms and tools that have been developed and tested in the field of microbial data analysis, but presents them in a more accessible, notebook-based format that's easier to use and deploy. As part of SIS's mission to support the scientific community, Biscuit provides researchers with powerful yet accessible tools for microscopy image analysis.

## **Migration Notes:**
- All Git history from the original project has been removed for fresh start
- Repository structure optimized for GitHub Pages deployment
- Focus shifted from GUI applications to notebook-based analysis
- Maintained compatibility with original segmentation algorithms


