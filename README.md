# 🧬 Spatial Transcriptomics Graph Neural Networks Demo

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.4+-orange.svg)](https://pytorch-geometric.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **An interactive educational platform bridging computational biology and graph neural networks for spatial cancer research**

## 🎯 What is This?

This comprehensive Streamlit application demonstrates how **Graph Neural Networks (GNNs)** revolutionize spatial transcriptomics analysis for cancer research. Perfect for bioinformaticians, computational biologists, and data scientists transitioning into spatial AI methods.

### 🧬 **For Biologists**: Learn how GNNs capture tissue architecture and cell communication patterns
### 🔢 **For Computer Scientists**: Master the mathematical foundations and implementation details

---

## ✨ Key Features

| 🔬 **Biological Track** | 🔢 **Mathematical Track** |
|-------------------------|---------------------------|
| 🧬 Cell communication analogies | 📐 Spectral graph theory foundations |
| 💊 Drug discovery applications | ⚡ Computational complexity analysis |
| 🏥 Clinical translation pathways | 🎯 Optimization frameworks |
| 🔬 Spatial biomarker discovery | 📊 Statistical validation methods |

## 🏗️ Architecture Comparison

Explore and compare four major GNN architectures:

- **🌐 Graph Convolutional Networks (GCN)**: Democratic cell communication
- **🎯 Graph Attention Networks (GAT)**: Selective neighbor focus  
- **📊 GraphSAGE**: Inductive learning across samples
- **🚀 Graph Transformers**: Global attention mechanisms

Each architecture includes:
- Interactive parameter exploration
- Real-time visualization 
- Performance comparisons
- Implementation guidance

---

## 🚀 Quick Start

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MateuszJakiel/GCN-Intro.git
   cd GCN-Intro
   ```

2. **Run the auto-setup launcher**
   ```bash
   chmod +x launch.sh
   ./launch.sh
   ```

The launcher automatically:
- ✅ Creates conda environment with all dependencies
- 📦 Installs PyTorch Geometric and spatial analysis tools
- 🔍 Verifies installation integrity
- 🌐 Opens the app in your browser at `http://localhost:8501`

### Manual Installation (Alternative)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate spatial-gcn-demo

# Launch app
streamlit run spatial_gcn_demo.py
```

---

## 📚 Learning Journey

### 🎯 **Phase 1: Overview**
- Why spatial context matters in cancer biology
- Graph neural networks vs traditional methods
- Mathematical foundations (spectral graph theory)

### 📊 **Phase 2: Data Architecture** 
- Spatial transcriptomics data structures
- Interactive tissue region exploration
- Statistical properties and preprocessing

### 🕸️ **Phase 3: Graph Construction**
- From coordinates to graph topology
- Neighborhood definition strategies
- Biological interpretation of graph properties

### 🧠 **Phase 4: GNN Architectures**
- Step-by-step message passing explanation
- Comparative analysis of four major architectures
- Implementation details and trade-offs

### ⚖️ **Phase 5: Architecture Comparison**
- Performance benchmarking
- Use case decision trees
- Mathematical complexity analysis

### 🔬 **Phase 6: Training & Analysis**
- Model training and validation
- Layer-wise feature evolution
- Interpretability and attention analysis

### 💡 **Phase 7: Real-World Applications**
- Drug target discovery pipelines
- Spatial biomarker development
- Treatment response prediction
- Clinical translation pathways

---

## 🎓 Educational Tracks

### 🧬 **Biology-Focused Track**
Perfect for:
- Pharmaceutical researchers
- Clinical scientists  
- Computational biologists
- Cancer researchers

**Focus**: Biological intuition, medical applications, clinical translation

### 🔢 **Mathematics-Focused Track**  
Perfect for:
- Computer scientists
- Machine learning engineers
- Applied mathematicians
- Algorithm developers

**Focus**: Mathematical rigor, implementation details, optimization theory

---

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **ML Framework** | PyTorch + PyTorch Geometric | Graph neural networks |
| **Data Science** | pandas, NumPy, scikit-learn | Data processing & analysis |
| **Visualization** | Plotly, matplotlib, seaborn | Interactive plots & graphs |
| **Graph Analysis** | NetworkX | Graph algorithms & metrics |
| **Statistics** | SciPy, statsmodels | Statistical testing & validation |

---

## 📖 Key Concepts Covered

### 🔬 **Biological Concepts**
- Tumor microenvironment architecture
- Cell-cell communication mechanisms
- Spatial gene expression patterns
- Treatment resistance mechanisms
- Biomarker discovery principles

### 🔢 **Mathematical Concepts**
- Spectral graph theory
- Message passing neural networks
- Attention mechanisms
- Inductive vs transductive learning
- Statistical validation frameworks

### 💻 **Implementation Skills**
- Graph construction from spatial data
- GNN architecture comparison
- Performance evaluation metrics
- Interpretability analysis
- Clinical validation pipelines

---

## 🌟 Real-World Applications

### 🎯 **Drug Discovery**
- Spatial-specific target identification
- Off-target effect prediction
- Combination therapy design

### 🔬 **Biomarker Development**  
- Spatial signature discovery
- Treatment response prediction
- Prognostic model development

### 🏥 **Clinical Translation**
- Regulatory approval pathways
- Clinical workflow integration
- Health economics evaluation

### 📊 **Research Tools**
- Multi-patient analysis pipelines
- Cross-platform validation
- Reproducible research frameworks

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution:
- 🧬 Additional biological use cases
- 🔢 Advanced mathematical methods
- 🏗️ New GNN architectures
- 📊 Validation datasets
- 📚 Educational content

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built for the computational biology community to bridge the gap between graph neural networks and spatial transcriptomics analysis.

**Special thanks to:**
- PyTorch Geometric team for excellent graph ML tools
- Streamlit team for the amazing interactive platform  
- Spatial transcriptomics community for pioneering this field

---

<div align="center">

---

*Made with ❤️ for the spatial biology community*

</div>