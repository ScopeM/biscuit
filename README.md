# 🍪 Biscuit

[![Paper DOI](https://img.shields.io/badge/DOI-10.12688/f1000research.171889.1-blue.svg)](https://doi.org/10.12688/f1000research.171889.1)

**Biscuit** is a powerful, notebook-based image segmentation tool designed for researchers and data scientists working with biological microscopy data. It provides an intuitive interface for cell segmentation through interactive Jupyter notebooks, making advanced image analysis accessible to everyone.

## 🚀 Features

- **Interactive Jupyter Notebooks** - Easy-to-use interface for image segmentation
- **Multiple Segmentation Algorithms** - Support for Cellpose, StarDist, Omnipose, and custom UNet models
- **Real-time Analysis** - Live visualization and results display
- **Cross-platform Compatibility** - Works on macOS, Linux, and Windows
- **GPU Acceleration** - Optimized for NVIDIA GPUs with TensorFlow support
- **Sample Data Included** - Ready-to-use example images for testing

## 📖 Usage

### Quick Start

5. **Open the notebooks:**
   - Google Colab notebook: https://colab.research.google.com/github/ScopeM/biscuit/blob/main/notebook/biscuit_google_colab.ipynb
   - `notebook/biscuit_eth_euler.ipynb` - ETH Euler cluster segmentation notebook (copy to the cluster an use it there)

### Basic Workflow

1. **Upload your images** to the data folder
2. **Select segmentation algorithm** (Cellpose, StarDist, Omnipose)
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

## 🎯 Use Cases

- **Microbial Cell Analysis** - Bacterial cell segmentation and tracking
- **Fluorescence Microscopy** - Analysis of fluorescently labeled cells
- **Time-lapse Imaging** - Processing of time-series data
- **Research Applications** - Academic and industrial research projects

### Custom Models

Biscuit supports custom trained models. Place your model weights in the appropriate directory and select them in the notebook interface.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scientific IT Services (SIS), ETH Zurich** - Project leadership and development
- **Szymon Stoma** - Technical leadership and contributions
- **Andrzej Rzepiela** - Technical leadership and contributions
- **Franziska Oschmann** - Project spearhead and vision
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

## 📚 Citation

If you use **BISCUIT** in your research, please cite:

**Rantsiou E, Oschmann F, von Ziegler L, Rzepiela, A J, and Stoma S**  
*BISCUIT: An Open-Source Platform for Visual Comparison of Segmentation Models in Bioimage Analysis*  
F1000Research 2025, 14:1277.  
https://doi.org/10.12688/f1000research.171889.1

<details>
<summary>BibTeX</summary>

<pre><code>@article{Rantsiou2025BISCUIT,
  title     = {BISCUIT: An Open-Source Platform for Visual Comparison of Segmentation Models in Bioimage Analysis},
  author    = {Rantsiou, E. and Oschmann, F. and von Ziegler, L. and Rzepiela, A. J. and Stoma, S.},
  journal   = {F1000Research},
  year      = {2025},
  volume    = {14},
  pages     = {1277},
  doi       = {10.12688/f1000research.171889.1}
}
</code></pre>

</details> 



## 🏛️ About SIS and ScopeM


**SIS** provides comprehensive IT services and expertise to the ETH Zurich research community, specializing in scientific computing, data analysis, and research software development. SIS spearheaded the BISCUIT project with the vision of creating accessible, powerful tools for biological image analysis.

**ScopeM** provides state-of-the-art microscopy facilities and expertise to the scientific community, supporting cutting-edge research in various fields including biology, materials science, and nanotechnology. Their practical contributions and domain expertise have been invaluable in shaping BISCUIT into a tool that meets real-world research needs.

**Made with ❤️ for the scientific community by SIS and ScopeM, ETH Zurich**
