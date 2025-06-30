#!/bin/bash

# Spatial Transcriptomics GNN Demo Launcher
# This script sets up the conda environment and launches the Streamlit app

set -e  # Exit on any error

echo "🧬 Spatial Transcriptomics GNN Demo Launcher"
echo "============================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Environment name
ENV_NAME="spatial-gcn-demo"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "✅ Environment '${ENV_NAME}' already exists"

    # Ask user if they want to update
    read -p "🔄 Do you want to update the environment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Updating environment..."
        conda env update -n ${ENV_NAME} -f environment.yml
    fi
else
    echo "🔧 Creating new environment '${ENV_NAME}'..."
    conda env create -f environment.yml
fi

# Activate environment and install PyG dependencies
echo "🚀 Activating environment and installing PyTorch Geometric..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install PyTorch Geometric with pip (more reliable for latest versions)
echo "📦 Installing PyTorch Geometric dependencies..."
pip install torch-geometric --quiet

# Verify key packages
echo "🔍 Verifying installation..."
python -c "
import streamlit
import torch
import torch_geometric
import pandas as pd
import numpy as np
import plotly
import networkx as nx
import sklearn
print('✅ All packages imported successfully!')
print(f'   - Streamlit: {streamlit.__version__}')
print(f'   - PyTorch: {torch.__version__}')
print(f'   - PyTorch Geometric: {torch_geometric.__version__}')
print(f'   - Pandas: {pd.__version__}')
print(f'   - NumPy: {np.__version__}')
"

# Check if the Python script exists
if [ ! -f "spatial_gcn_demo.py" ]; then
    echo "❌ Error: spatial_gcn_demo.py not found in current directory"
    echo "Please ensure the Python script is in the same directory as this launcher"
    exit 1
fi

echo ""
echo "🎉 Setup complete! Launching Streamlit app..."
echo ""
echo "📖 Usage Instructions:"
echo "   • Choose your learning track (Biology or Mathematics)"
echo "   • Navigate through sections using the sidebar"
echo "   • Interact with parameters to see real-time results"
echo "   • Switch tracks anytime to see different explanations"
echo ""
echo "🌐 The app will open in your default browser"
echo "   • Local URL: http://localhost:8501"
echo "   • Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit with optimized settings
streamlit run spatial_gcn_demo.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless false \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#1f77b4" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f0f2f6"