# 🍪 Biscuit

**Biscuit** is a powerful, notebook-based image segmentation tool designed for researchers and data scientists working with biological microscopy data. It provides an intuitive interface for cell segmentation through interactive Jupyter notebooks, making advanced image analysis accessible to everyone.

## 🚀 Features

- **Interactive Jupyter Notebooks** - Easy-to-use interface for image segmentation
- **Multiple Segmentation Algorithms** - Support for Cellpose, StarDist, Omnipose, and custom UNet models
- **Real-time Analysis** - Live visualization and results display
- **Cross-platform Compatibility** - Works on macOS, Linux, and Windows
- **GPU Acceleration** - Optimized for NVIDIA GPUs with TensorFlow support
- **Sample Data Included** - Ready-to-use example images for testing

## 📋 Requirements

- **Python 3.9+**
- **Jupyter Notebook or JupyterLab**
- **NVIDIA GPU** (optional, for acceleration)
- **8GB+ RAM** (recommended)

## 🛠️ Installation

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ScopeM/biscuit.git
   cd biscuit
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv biscuit-env
   source biscuit-env/bin/activate  # On Windows: biscuit-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install jupyter numpy scipy scikit-image opencv-python pandas matplotlib ipywidgets
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

5. **Open the notebooks:**
   - `notebook/biscuit_google_colab.ipynb` - Google Colab compatible segmentation notebook
   - `notebook/biscuit_eth_euler.ipynb` - ETH Euler cluster segmentation notebook

## 📖 Usage

### Basic Workflow

1. **Upload your images** to the data folder
2. **Select segmentation algorithm** (Cellpose, StarDist, Omnipose, or UNet)
3. **Adjust parameters** as needed
4. **Run segmentation** and view results
5. **Export results** in various formats

### Supported Image Formats

- TIFF/TIF (recommended)
- PNG
- JPEG
- OME-TIFF

### Segmentation Algorithms

- **Cellpose** - General-purpose cell segmentation
- **StarDist** - Star-convex object detection
- **Omnipose** - Advanced cell segmentation with improved boundary detection
- **UNet** - Custom neural network models

## 🎯 Use Cases

- **Microbial Cell Analysis** - Bacterial cell segmentation and tracking
- **Fluorescence Microscopy** - Analysis of fluorescently labeled cells
- **Time-lapse Imaging** - Processing of time-series data
- **Research Applications** - Academic and industrial research projects

## 🔧 Advanced Configuration

### GPU Support

For GPU acceleration, install TensorFlow with GPU support:

```bash
pip install tensorflow[gpu]
```

### Custom Models

Biscuit supports custom trained models. Place your model weights in the appropriate directory and select them in the notebook interface.

## 📊 Performance

- **Processing Speed:** 10-100 frames per minute (depending on image size and GPU)
- **Memory Usage:** 2-8GB RAM (depending on image stack size)
- **Accuracy:** Comparable to state-of-the-art segmentation tools

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scientific IT Services (SIS), ETH Zurich** - Project leadership and development
- **Franziska Oschmann** - Project spearhead and vision
- **Szymon Stoma** - Technical leadership and contributions
- **Andrzej Rzepiela** - Technical leadership and contributions
- **ScopeM, ETH Zurich** - Collaboration and domain expertise
- **Cellpose Team** - Cellpose algorithm
- **StarDist Team** - StarDist algorithm
- **Omnipose Team** - Omnipose algorithm

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/ScopeM/biscuit/issues)
- **Documentation:** [Wiki](https://github.com/ScopeM/biscuit/wiki)
- **Email:** [biscuit@scopem.ethz.ch]

## 🔗 Links

- **Website:** [http://biscuit.let-your-data-speak.com/](http://biscuit.let-your-data-speak.com/)
- **Documentation:** [GitHub Wiki](https://github.com/ScopeM/biscuit/wiki)
- **Releases:** [GitHub Releases](https://github.com/ScopeM/biscuit/releases)

---

## 📜 Project History

**BISCUIT (BioImage Segmentation Comparison Utility and Interactive Tool)** traces its roots back to the **Scientific IT Services (SIS)** at ETH Zurich. Spearheaded by **Franziska Oschmann**, the goal was to create an intuitive yet powerful platform for visually comparing cell segmentation models on microscopy data.

The initial version of BISCUIT was built on top of the **Microbial Image Data Analysis Pipeline (MIDAP)** framework—a modular system originally developed for analyzing data from Mother Machine experiments. MIDAP's flexible architecture made it a solid foundation for building interactive segmentation workflows.

Development continued in close collaboration with the ETH Zurich imaging center **ScopeM**, with practical contributions and technical leadership by **Szymon Stoma** and **Andrzej Rzepiela**. Together, the team laid the groundwork for a tool that aspires to become a versatile visual benchmarking suite for segmentation models in the Life Sciences.

Rather than creating a traditional fork, we decided to create a clean, optimized copy that focuses specifically on the core segmentation functionality through Jupyter notebooks. This approach allows us to maintain a focused codebase while preserving the powerful segmentation algorithms that have proven effective in biological image analysis.

## 🏛️ About SIS and ScopeM

**BISCUIT** is developed by the **Scientific IT Services (SIS)** at ETH Zurich in close collaboration with **[ScopeM](https://scopem.ethz.ch/)**, the Scientific Center for Optical and Electron Microscopy at ETH Zurich.

**SIS** provides comprehensive IT services and expertise to the ETH Zurich research community, specializing in scientific computing, data analysis, and research software development. SIS spearheaded the BISCUIT project with the vision of creating accessible, powerful tools for biological image analysis.

**ScopeM** provides state-of-the-art microscopy facilities and expertise to the scientific community, supporting cutting-edge research in various fields including biology, materials science, and nanotechnology. Their practical contributions and domain expertise have been invaluable in shaping BISCUIT into a tool that meets real-world research needs.

**Made with ❤️ for the scientific community by SIS and ScopeM, ETH Zurich**
