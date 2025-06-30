import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.data import Data
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Spatial Transcriptomics GNN Comprehensive Demo",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .biological-insight {
        background-color: #e8f5e8;
        padding: 1rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .math-insight {
        background-color: #fff3e0;
        padding: 1rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .cs-concept {
        background-color: #e8f4fd;
        padding: 1rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .math-box {
        background-color: #fff3cd;
        padding: 1rem;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .track-selector {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🧬 Comprehensive Spatial Transcriptomics with Graph Neural Networks</h1>',
            unsafe_allow_html=True)

st.markdown("""
**A dual-track interactive exploration bridging biology and mathematics in spatial AI for cancer research**

Choose your learning track below to get explanations tailored to your background and interests.
""")

# Track Selection
st.markdown('<div class="track-selector">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 🎯 Choose Your Learning Track")
    track = st.radio(
        "",
        ["🧬 Biology-Focused Track", "🔢 Mathematics-Focused Track"],
        help="Choose based on your background and learning goals"
    )

if track == "🧬 Biology-Focused Track":
    st.markdown("""
    **Perfect for:** Biologists, pharmaceutical researchers, clinicians transitioning to computational methods

    **Focus:** Biological intuition, medical applications, interpretation of results for discovery
    """)
else:
    st.markdown("""
    **Perfect for:** Computer scientists, mathematicians, engineers entering spatial biology

    **Focus:** Mathematical foundations, algorithmic details, computational complexity, implementation
    """)

st.markdown('</div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("🔬 Navigation")
sections = [
    "🎯 Overview",
    "📊 Data Structure",
    "🕸️ Graph Construction",
    "🧠 GNN Architectures",
    "⚖️ Architecture Comparison",
    "🔬 Training & Analysis",
    "💡 Applications & Insights"
]
section = st.sidebar.radio("Choose Section:", sections)


# Helper functions
@st.cache_data
def generate_spatial_transcriptomics_data():
    """Generate realistic spatial transcriptomics data resembling tumor tissue"""
    np.random.seed(42)
    n_spots = 200

    # Create spatial coordinates (mimicking Visium array layout)
    x = np.random.uniform(0, 20, n_spots)
    y = np.random.uniform(0, 20, n_spots)

    # Define tissue regions with biological meaning
    regions = []
    region_names = []

    for i in range(n_spots):
        # Tumor core (center, high proliferation genes)
        if (x[i] - 10) ** 2 + (y[i] - 10) ** 2 < 16:
            regions.append(0)
            region_names.append("Tumor Core")
        # Tumor edge (intermediate region)
        elif (x[i] - 10) ** 2 + (y[i] - 10) ** 2 < 36:
            regions.append(1)
            region_names.append("Tumor Edge")
        # Immune infiltrate (scattered)
        elif np.random.random() < 0.3:
            regions.append(2)
            region_names.append("Immune Zone")
        # Normal tissue
        else:
            regions.append(3)
            region_names.append("Normal Tissue")

    # Generate gene expression based on regions
    n_genes = 50
    gene_names = [f"Gene_{i + 1}" for i in range(n_genes)]

    # Key cancer-related gene signatures
    proliferation_genes = ['MKI67', 'PCNA', 'TOP2A', 'CCNB1', 'CDK1']
    immune_genes = ['CD3D', 'CD8A', 'IFNG', 'IL2', 'GZMB']
    stress_genes = ['HSP90AA1', 'HSPA1A', 'JUN', 'FOS', 'ATF3']
    normal_genes = ['ALB', 'TTR', 'APOE', 'SERPINA1', 'TF']

    # Replace generic names with meaningful ones
    gene_names[:5] = proliferation_genes
    gene_names[5:10] = immune_genes
    gene_names[10:15] = stress_genes
    gene_names[15:20] = normal_genes

    expression_data = np.zeros((n_spots, n_genes))

    for i in range(n_spots):
        region = regions[i]

        if region == 0:  # Tumor core - high proliferation
            expression_data[i, :5] = np.random.lognormal(3, 0.5, 5)  # High proliferation genes
            expression_data[i, 5:10] = np.random.lognormal(1, 0.3, 5)  # Low immune genes
            expression_data[i, 10:15] = np.random.lognormal(2.5, 0.4, 5)  # High stress genes
            expression_data[i, 15:] = np.random.lognormal(1.5, 0.3, n_genes - 15)

        elif region == 1:  # Tumor edge - mixed signals
            expression_data[i, :5] = np.random.lognormal(2.2, 0.4, 5)
            expression_data[i, 5:10] = np.random.lognormal(2, 0.4, 5)
            expression_data[i, 10:15] = np.random.lognormal(2.2, 0.3, 5)
            expression_data[i, 15:] = np.random.lognormal(1.8, 0.3, n_genes - 15)

        elif region == 2:  # Immune zone - high immune genes
            expression_data[i, :5] = np.random.lognormal(1.2, 0.3, 5)
            expression_data[i, 5:10] = np.random.lognormal(3.2, 0.5, 5)  # High immune genes
            expression_data[i, 10:15] = np.random.lognormal(1.8, 0.3, 5)
            expression_data[i, 15:] = np.random.lognormal(1.5, 0.3, n_genes - 15)

        else:  # Normal tissue
            expression_data[i, :5] = np.random.lognormal(0.5, 0.2, 5)
            expression_data[i, 5:10] = np.random.lognormal(0.8, 0.2, 5)
            expression_data[i, 10:15] = np.random.lognormal(1, 0.2, 5)
            expression_data[i, 15:20] = np.random.lognormal(2.5, 0.4, 5)  # High normal genes
            expression_data[i, 20:] = np.random.lognormal(1.2, 0.3, n_genes - 20)

    # Add noise
    expression_data += np.random.normal(0, 0.1, expression_data.shape)
    expression_data = np.maximum(expression_data, 0)  # Ensure non-negative

    return pd.DataFrame({
        'x': x,
        'y': y,
        'region': regions,
        'region_name': [region_names[i] for i in regions]
    }), pd.DataFrame(expression_data, columns=gene_names)


def create_spatial_graph(coords, k_neighbors=6, distance_threshold=2.0):
    """Create spatial graph from coordinates"""
    from sklearn.neighbors import NearestNeighbors

    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # Create edge list
    edges = []
    edge_weights = []

    for i in range(len(coords)):
        for j in range(1, len(indices[i])):  # Skip self
            neighbor_idx = indices[i][j]
            distance = distances[i][j]

            if distance <= distance_threshold:
                edges.append([i, neighbor_idx])
                edge_weights.append(1.0 / (1.0 + distance))  # Weight inversely proportional to distance

    return np.array(edges).T, edge_weights


# Different GNN architectures
class SpatialGCN(nn.Module):
    """Graph Convolutional Network for spatial transcriptomics"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(SpatialGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, self.dropout, training=self.training)

        x3 = self.conv3(x2, edge_index)

        return x3, x2, x1


class SpatialGAT(nn.Module):
    """Graph Attention Network for spatial transcriptomics"""

    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.1):
        super(SpatialGAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, self.dropout, training=self.training)

        x3 = self.conv3(x2, edge_index)

        return x3, x2, x1


class SpatialSAGE(nn.Module):
    """GraphSAGE for spatial transcriptomics"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(SpatialSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, self.dropout, training=self.training)

        x3 = self.conv3(x2, edge_index)

        return x3, x2, x1


class SpatialTransformer(nn.Module):
    """Graph Transformer for spatial transcriptomics"""

    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.1):
        super(SpatialTransformer, self).__init__()
        self.conv1 = TransformerConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.conv3 = TransformerConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, self.dropout, training=self.training)

        x3 = self.conv3(x2, edge_index)

        return x3, x2, x1


# Load data
coords_df, expression_df = generate_spatial_transcriptomics_data()


# Track-specific content functions
def show_biology_content(content_type, **kwargs):
    """Show biology-focused content"""
    if content_type == "overview_intro":
        return """
        <div class="biological-insight">
        <h4>🧬 Why Spatial Matters in Cancer Biology</h4>

        Imagine studying a city by randomly interviewing people without knowing where they live. You'd miss crucial insights about neighborhoods, social interactions, and local environments.

        <strong>Traditional RNA-seq has the same problem:</strong> it tells us what genes are active but not WHERE in the tissue they're active.

        <strong>In cancer, location is everything:</strong>
        • Cancer cells behave differently when surrounded by immune cells vs. other cancer cells
        • Drug resistance often emerges in specific tissue regions
        • Metastasis starts from particular spatial niches

        <strong>Spatial transcriptomics preserves this crucial context.</strong>
        </div>
        """

    elif content_type == "graph_explanation":
        return """
        <div class="biological-insight">
        <h4>🕸️ Graphs as Biological Networks</h4>

        <strong>Think of tissue as a cellular social network:</strong>
        • Each tissue spot = a person (node)
        • Physical proximity = friendship connections (edges)
        • Gene expression = personality traits (features)
        • Goal: understand how neighbors influence each other

        <strong>Biological principle:</strong> Cells communicate through chemical signals that work over short distances. Neighboring cells have more influence on each other than distant ones.

        This is why graph structure captures biology better than treating cells as independent entities.
        </div>
        """


def show_math_content(content_type, **kwargs):
    """Show mathematics-focused content"""
    if content_type == "overview_intro":
        return """
        <div class="math-insight">
        <h4>🔢 Mathematical Formulation of Spatial Transcriptomics</h4>

        <strong>Problem Setup:</strong>
        • Graph G = (V, E) where V = tissue spots, E = spatial adjacency
        • Node features X ∈ ℝ^{n×d} where n = spots, d = genes
        • Spatial coordinates C ∈ ℝ^{n×2}
        • Goal: Learn f: (X, G) → Z where Z captures spatial patterns

        <strong>Key Challenge:</strong> Standard CNNs assume grid structure (images), but tissue has irregular geometry. We need convolution operations that work on arbitrary graphs.

        <strong>Solution:</strong> Spectral graph theory provides the mathematical foundation for generalizing convolution to graphs through the graph Laplacian eigenbasis.
        </div>
        """

    elif content_type == "graph_math":
        return """
        <div class="math-insight">
        <h4>🔢 Graph Construction Mathematics</h4>

        <strong>Adjacency Matrix Construction:</strong>

        Given coordinates C ∈ ℝ^{n×2}, construct A ∈ {0,1}^{n×n}:

        A_{ij} = 1 if ||C_i - C_j||_2 ≤ τ and i ≠ j, else 0

        <strong>Degree Matrix:</strong> D_{ii} = Σ_j A_{ij}

        <strong>Normalized Laplacian:</strong> L_norm = I - D^{-1/2}AD^{-1/2}

        <strong>Properties:</strong>
        • Symmetric: L_norm = L_norm^T
        • Eigenvalues: 0 = λ_1 ≤ λ_2 ≤ ... ≤ λ_n ≤ 2
        • Eigenvectors form orthonormal basis for graph signals

        <strong>Computational Complexity:</strong> O(nk) for k-NN construction
        </div>
        """


# Section: Overview
if section == "🎯 Overview":
    st.markdown('<div class="section-header">Understanding Spatial Biology with Graph Neural Networks</div>',
                unsafe_allow_html=True)

    if track == "🧬 Biology-Focused Track":
        st.markdown(show_biology_content("overview_intro"), unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div class="cs-concept">
            <h4>💻 Why Graph Neural Networks?</h4>

            <strong>The Challenge:</strong> How do we teach computers to understand tissue architecture?

            <strong>The Solution:</strong> Represent tissue as a graph where:
            • Nodes = tissue spots (with gene expression data)
            • Edges = spatial relationships ("who's neighbors with whom")
            • Neural networks learn from this structured data

            <strong>The Magic:</strong> GNNs can find patterns like "cancer cells near immune cells express different stress genes" automatically from the data.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            **Real-world Applications:**
            - 🎯 **Drug Target Discovery**: Find genes expressed only in tumor regions
            - 🔬 **Biomarker Identification**: Discover spatial signatures that predict treatment response  
            - 💊 **Personalized Medicine**: Tailor treatments based on spatial tumor architecture
            - 🧬 **Basic Research**: Understand how cells communicate in tissue
            """)

        with col2:
            # Simple workflow diagram
            fig = go.Figure()

            steps = [
                {"name": "Tissue\nSection", "x": 0, "y": 3, "color": "#ff6b6b"},
                {"name": "Spatial\nTranscriptomics", "x": 1, "y": 3, "color": "#4ecdc4"},
                {"name": "Graph\nConstruction", "x": 2, "y": 3, "color": "#45b7d1"},
                {"name": "Graph Neural\nNetwork", "x": 3, "y": 3, "color": "#96ceb4"},
                {"name": "Biological\nInsights", "x": 4, "y": 3, "color": "#ffeaa7"}
            ]

            for i, step in enumerate(steps):
                fig.add_trace(go.Scatter(
                    x=[step["x"]], y=[step["y"]],
                    mode='markers+text',
                    marker=dict(size=60, color=step["color"]),
                    text=step["name"],
                    textposition="middle center",
                    textfont=dict(size=10, color="white"),
                    showlegend=False
                ))

                if i < len(steps) - 1:
                    fig.add_annotation(
                        x=step["x"] + 0.3, y=step["y"],
                        ax=step["x"] + 0.7, ay=step["y"],
                        arrowhead=2, arrowsize=1, arrowwidth=2,
                        showarrow=True
                    )

            fig.update_layout(
                title="From Tissue to Discovery",
                height=300,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[2, 4])
            )

            st.plotly_chart(fig, use_container_width=True)

    else:  # Math track
        st.markdown(show_math_content("overview_intro"), unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div class="math-box">
            <h4>📐 Core Mathematical Concepts</h4>

            <strong>Spectral Graph Theory:</strong>
            • Graph Laplacian L = D - A
            • Eigendecomposition L = UΛU^T
            • Graph Fourier Transform: x̂ = U^T x

            <strong>Graph Convolution:</strong>
            g_θ ★ x = U g_θ(Λ) U^T x

            <strong>Computational Bottleneck:</strong>
            • Eigendecomposition: O(n³)
            • For large graphs: prohibitive

            <strong>Kipf & Welling Innovation:</strong>
            First-order Chebyshev approximation → O(|E|)
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            **Mathematical Challenges:**
            - 🔢 **Non-Euclidean Data**: Graphs don't have regular grid structure
            - ⚡ **Scalability**: Spectral methods are O(n³) 
            - 🎯 **Localization**: Global spectral filters aren't localized
            - 🔄 **Permutation Invariance**: Node ordering shouldn't matter
            """)

        with col2:
            # Mathematical progression diagram
            st.markdown("""
            **Mathematical Evolution:**

            ```
            Euclidean Convolution (Images)
                      ↓
            Spectral Graph Convolution  
                      ↓
            Chebyshev Approximation
                      ↓
            First-Order (GCN)
                      ↓
            Attention Mechanisms (GAT)
                      ↓
            Transformer Architectures
            ```

            **Complexity Analysis:**
            - **Spectral Methods**: O(n³ + |E|)
            - **ChebNet**: O(K|E|) where K = polynomial order
            - **GCN**: O(|E|) - linear in edges!
            - **GAT**: O(|E| + n·h) where h = attention heads
            """)

# Section: Data Structure
elif section == "📊 Data Structure":
    st.markdown('<div class="section-header">Spatial Transcriptomics Data Architecture</div>', unsafe_allow_html=True)

    if track == "🧬 Biology-Focused Track":
        st.markdown("""
        <div class="biological-insight">
        <h4>🔬 Understanding the Tumor Microenvironment</h4>

        Our dataset models a realistic tumor tissue section with four distinct biological regions:

        • <strong>Tumor Core:</strong> Hypoxic center with high proliferation (MKI67, PCNA) and stress responses
        • <strong>Tumor Edge:</strong> The invasive front where cancer cells interact with normal tissue
        • <strong>Immune Infiltrate:</strong> T cells and other immune cells (CD3D, CD8A) attempting to fight cancer
        • <strong>Normal Tissue:</strong> Healthy cells with normal metabolic function (ALB, TTR)

        <strong>Why this matters:</strong> Each region requires different therapeutic approaches. Understanding spatial organization helps predict drug response and resistance.
        </div>
        """, unsafe_allow_html=True)

    else:  # Math track
        st.markdown("""
        <div class="math-insight">
        <h4>🔢 Data Structure and Representation</h4>

        <strong>Spatial Transcriptomics Data Tensor:</strong>
        • Coordinates: C ∈ ℝ^{n×2} (x, y positions)
        • Expression: X ∈ ℝ^{n×d} (n spots, d genes)
        • Labels: Y ∈ {0,1,2,3}^n (tissue regions)

        <strong>Data Generation Model:</strong>
        X_{ij} ~ LogNormal(μ_r(j), σ²) where r = region(i)

        <strong>Statistical Properties:</strong>
        • E[X_{ij}] depends on tissue region
        • Spatial autocorrelation in expression
        • Heavy-tailed distributions (log-normal)

        <strong>Preprocessing Requirements:</strong>
        • Log-transformation: X' = log(X + 1)
        • Standardization: X'' = (X' - μ)/σ
        • Dimensionality: Often d >> n (high-dimensional)
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📍 Spatial Layout")

        # Spatial plot
        fig = px.scatter(
            coords_df, x='x', y='y', color='region_name',
            title="Tissue Section Spatial Organization",
            color_discrete_map={
                'Tumor Core': '#d62728',
                'Tumor Edge': '#ff7f0e',
                'Immune Zone': '#2ca02c',
                'Normal Tissue': '#1f77b4'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        if track == "🧬 Biology-Focused Track":
            st.info(
                "💡 Notice the concentric organization - this mimics real tumor architecture where cell types organize in spatial layers")
        else:
            st.info("📐 Spatial coordinates follow uniform distribution with region-dependent probability assignments")

    with col2:
        st.subheader("🧬 Gene Expression Heatmap")

        # Show expression heatmap for key genes
        key_genes = ['MKI67', 'PCNA', 'CD3D', 'CD8A', 'ALB', 'TTR', 'HSPA1A', 'JUN']
        key_expression = expression_df[key_genes]

        fig = px.imshow(
            key_expression.T,
            title="Key Gene Expression Signatures",
            labels=dict(x="Tissue Spots", y="Genes", color="Expression Level"),
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Advanced data exploration
    st.subheader("🔬 Interactive Data Exploration")

    tab1, tab2, tab3 = st.tabs(["Expression Analysis", "Spatial Patterns", "Statistical Properties"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            if track == "🧬 Biology-Focused Track":
                st.markdown("**Data Overview:**")
                st.metric("Tissue Spots", len(coords_df))
                st.metric("Genes Measured", len(expression_df.columns))
                st.metric("Tumor Spots", len(coords_df[coords_df['region'] <= 1]))
                st.metric("Immune Spots", len(coords_df[coords_df['region'] == 2]))
            else:
                st.markdown("**Dataset Statistics:**")
                st.metric("Sample Size (n)", len(coords_df))
                st.metric("Feature Dimension (d)", len(expression_df.columns))
                st.metric("Sparsity",
                          f"{(expression_df == 0).sum().sum() / (len(coords_df) * len(expression_df.columns)):.2%}")
                st.metric("Mean Expression", f"{expression_df.values.mean():.2f}")

        with col2:
            selected_gene = st.selectbox("Select gene to analyze:",
                                         ['MKI67', 'CD3D', 'ALB', 'HSPA1A'] + list(expression_df.columns[:20]))

            # Expression distribution
            fig = px.histogram(
                expression_df, x=selected_gene,
                title=f"Expression Distribution: {selected_gene}",
                nbins=30
            )
            if track == "🔢 Mathematics-Focused Track":
                # Add statistical annotations
                mean_expr = expression_df[selected_gene].mean()
                std_expr = expression_df[selected_gene].std()
                fig.add_vline(x=mean_expr, line_dash="dash", annotation_text=f"μ={mean_expr:.2f}")
                fig.add_vline(x=mean_expr + std_expr, line_dash="dot", annotation_text=f"μ+σ")
                fig.add_vline(x=mean_expr - std_expr, line_dash="dot", annotation_text=f"μ-σ")

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        selected_gene = st.selectbox("Gene for spatial analysis:",
                                     ['MKI67', 'CD3D', 'ALB', 'HSPA1A'], key="spatial_gene")

        # Create spatial expression map
        plot_df = coords_df.copy()
        plot_df['expression'] = expression_df[selected_gene]

        fig = px.scatter(
            plot_df, x='x', y='y', color='expression',
            title=f"Spatial Expression Pattern: {selected_gene}",
            color_continuous_scale='Viridis',
            size='expression' if track == "🧬 Biology-Focused Track" else None
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        if track == "🧬 Biology-Focused Track":
            st.markdown(f"""
            <div class="biological-insight">
            <strong>Biological Interpretation:</strong> Notice how {selected_gene} shows distinct spatial clustering. 
            This pattern reflects the underlying biology - cells in similar microenvironments express similar genes.
            Graph neural networks will learn to capture and amplify these spatial patterns.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Calculate spatial autocorrelation
            from scipy.spatial.distance import pdist, squareform

            coords_array = coords_df[['x', 'y']].values
            distance_matrix = squareform(pdist(coords_array))

            # Moran's I calculation (simplified)
            weights = np.exp(-distance_matrix / 2.0)  # Gaussian weights
            weights[distance_matrix > 3.0] = 0  # Threshold
            np.fill_diagonal(weights, 0)

            gene_values = expression_df[selected_gene].values
            moran_i = np.sum(
                weights * np.outer(gene_values - gene_values.mean(), gene_values - gene_values.mean())) / np.sum(
                weights)
            moran_i /= np.var(gene_values)

            st.markdown(f"""
            <div class="math-insight">
            <strong>Spatial Autocorrelation Analysis:</strong>

            Moran's I = {moran_i:.3f}

            • I > 0: Positive spatial autocorrelation (clustering)
            • I ≈ 0: Random spatial pattern  
            • I < 0: Negative spatial autocorrelation (checkerboard)

            This quantifies how much spatial location determines expression.
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        if track == "🧬 Biology-Focused Track":
            # Compare expression across regions
            plot_data = []
            for region in coords_df['region'].unique():
                region_mask = coords_df['region'] == region
                region_name = coords_df[region_mask]['region_name'].iloc[0]
                for gene in ['MKI67', 'CD3D', 'ALB', 'HSPA1A']:
                    plot_data.append({
                        'Region': region_name,
                        'Gene': gene,
                        'Expression': expression_df.loc[region_mask, gene].mean()
                    })

            plot_df = pd.DataFrame(plot_data)
            fig = px.bar(
                plot_df, x='Gene', y='Expression', color='Region',
                title="Average Gene Expression by Tissue Region",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="biological-insight">
            <strong>Key Observation:</strong> Different tissue regions have distinct expression profiles.
            This is the biological signal that graph neural networks will learn to amplify and use for classification.
            </div>
            """, unsafe_allow_html=True)

        else:
            # Statistical analysis
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Distribution Properties:**")

                # Calculate skewness and kurtosis
                from scipy import stats

                stats_data = []
                for gene in expression_df.columns[:10]:
                    gene_expr = expression_df[gene]
                    stats_data.append({
                        'Gene': gene,
                        'Mean': gene_expr.mean(),
                        'Std': gene_expr.std(),
                        'Skewness': stats.skew(gene_expr),
                        'Kurtosis': stats.kurtosis(gene_expr)
                    })

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df.round(3), use_container_width=True)

            with col2:
                st.markdown("**Correlation Structure:**")

                # Compute correlation matrix for subset of genes
                key_genes = ['MKI67', 'CD3D', 'ALB', 'HSPA1A', 'JUN', 'FOS']
                corr_matrix = expression_df[key_genes].corr()

                fig = px.imshow(
                    corr_matrix,
                    title="Gene-Gene Correlation Matrix",
                    color_continuous_scale="RdBu",
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)

# Section: Graph Construction
elif section == "🕸️ Graph Construction":
    st.markdown('<div class="section-header">From Spatial Coordinates to Graph Structure</div>', unsafe_allow_html=True)

    if track == "🧬 Biology-Focused Track":
        st.markdown(show_biology_content("graph_explanation"), unsafe_allow_html=True)
    else:
        st.markdown(show_math_content("graph_math"), unsafe_allow_html=True)

    # Interactive graph construction
    st.subheader("🎛️ Interactive Graph Construction")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Graph Parameters:**")

        k_neighbors = st.slider("K-Nearest Neighbors", 3, 12, 6,
                                help="Number of closest spatial neighbors to connect")

        distance_threshold = st.slider("Distance Threshold", 1.0, 4.0, 2.0, 0.1,
                                       help="Maximum distance for edge connection")

        if track == "🧬 Biology-Focused Track":
            st.markdown("""
            <div class="biological-insight">
            <strong>Biological Intuition:</strong>

            • **K-neighbors**: How many cell neighbors can directly influence each cell through paracrine signaling
            • **Distance threshold**: Maximum range of cell-cell communication (typically ~100-200μm in tissue)

            Too few connections = cells seem isolated<br>
            Too many connections = distant cells inappropriately influence each other
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="math-box">
            <strong>Algorithm:</strong><br>
            1. ∀i: Find k nearest neighbors N(i)<br>
            2. Add edge (i,j) if j ∈ N(i) AND d(i,j) ≤ τ<br>
            3. Weight: w_{ij} = exp(-d(i,j)²/σ²)<br>
            4. Normalize: Ã = D^{-1/2}AD^{-1/2}

            <strong>Complexity:</strong> O(n log n) for k-NN<br>
            <strong>Memory:</strong> O(|E|) for sparse storage
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Construct and visualize graph
        edges, weights = create_spatial_graph(
            coords_df[['x', 'y']].values,
            k_neighbors=k_neighbors,
            distance_threshold=distance_threshold
        )

        # Visualize graph
        fig = go.Figure()

        # Add edges (sample for performance)
        edge_x, edge_y = [], []
        for i, edge in enumerate(edges.T):
            if i > 300:  # Limit for performance
                break
            x0, y0 = coords_df.iloc[edge[0]][['x', 'y']]
            x1, y1 = coords_df.iloc[edge[1]][['x', 'y']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.8, color='lightgray'),
            name='Spatial Connections',
            showlegend=False
        ))

        # Add nodes colored by region
        fig.add_trace(go.Scatter(
            x=coords_df['x'], y=coords_df['y'],
            mode='markers',
            marker=dict(
                color=coords_df['region'],
                colorscale='Viridis',
                size=8,
                line=dict(width=1, color='white')
            ),
            text=coords_df['region_name'],
            name='Tissue Spots'
        ))

        fig.update_layout(
            title=f"Spatial Graph Structure (Edges: {len(edges.T)}, Avg Degree: {2 * len(edges.T) / len(coords_df):.1f})",
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # Graph analysis
    st.subheader("📊 Graph Properties Analysis")

    # Create NetworkX graph for analysis
    G = nx.Graph()
    G.add_edges_from(edges.T)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Edges", len(edges.T))
    with col2:
        avg_degree = 2 * len(edges.T) / len(coords_df)
        st.metric("Average Degree", f"{avg_degree:.1f}")
    with col3:
        density = nx.density(G)
        st.metric("Graph Density", f"{density:.3f}")
    with col4:
        if nx.is_connected(G):
            diameter = nx.diameter(G)
            st.metric("Graph Diameter", diameter)
        else:
            components = nx.number_connected_components(G)
            st.metric("Connected Components", components)

    # Advanced graph analysis
    tab1, tab2, tab3 = st.tabs(["Degree Distribution", "Local Analysis", "Spectral Properties"])

    with tab1:
        degrees = [G.degree(n) for n in G.nodes()]

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                x=degrees, nbins=20,
                title="Node Degree Distribution",
                labels={'x': 'Degree', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if track == "🧬 Biology-Focused Track":
                st.markdown("""
                **Biological Interpretation:**

                • **High degree nodes**: Central locations with many neighbors (like hub cells in tissue)
                • **Low degree nodes**: Edge/boundary regions with fewer connections
                • **Distribution shape**: Tells us about tissue organization patterns

                Most biological networks show degree heterogeneity - some cells are more "connected" than others.
                """)
            else:
                mean_deg = np.mean(degrees)
                std_deg = np.std(degrees)
                st.markdown(f"""
                **Statistical Properties:**

                • Mean degree: {mean_deg:.2f}
                • Std deviation: {std_deg:.2f}
                • Coefficient of variation: {std_deg / mean_deg:.2f}

                **Implications for GNNs:**
                • High-degree nodes aggregate more information
                • Normalization prevents degree bias
                • Affects message passing dynamics
                """)

    with tab2:
        # Select node for local analysis
        example_node = st.selectbox("Select tissue spot for local analysis:",
                                    range(min(20, len(coords_df))))

        # Get neighbors
        neighbors = list(G.neighbors(example_node))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **Local Neighborhood Analysis:**
            - Selected spot: #{example_node}
            - Region: {coords_df.iloc[example_node]['region_name']}
            - Coordinates: ({coords_df.iloc[example_node]['x']:.1f}, {coords_df.iloc[example_node]['y']:.1f})
            - Number of neighbors: {len(neighbors)}
            """)

            if len(neighbors) > 0:
                # Local clustering coefficient
                subgraph = G.subgraph([example_node] + neighbors)
                local_clustering = nx.clustering(G, example_node)
                st.metric("Local Clustering Coefficient", f"{local_clustering:.3f}")

                if track == "🔢 Mathematics-Focused Track":
                    st.markdown(f"""
                    **Local Properties:**
                    - Clustering coefficient: {local_clustering:.3f}
                    - Neighborhood size: {len(neighbors)}
                    - Density of neighborhood: {nx.density(subgraph):.3f}
                    """)

        with col2:
            # Visualize local neighborhood
            fig = go.Figure()

            # Plot all spots in light gray
            fig.add_trace(go.Scatter(
                x=coords_df['x'], y=coords_df['y'],
                mode='markers',
                marker=dict(color='lightgray', size=6),
                name='Other Spots',
                showlegend=False
            ))

            # Highlight neighbors in orange
            if len(neighbors) > 0:
                neighbor_coords = coords_df.iloc[neighbors]
                fig.add_trace(go.Scatter(
                    x=neighbor_coords['x'], y=neighbor_coords['y'],
                    mode='markers',
                    marker=dict(color='orange', size=10),
                    name='Neighbors'
                ))

                # Draw edges to neighbors
                for neighbor in neighbors:
                    fig.add_trace(go.Scatter(
                        x=[coords_df.iloc[example_node]['x'], coords_df.iloc[neighbor]['x']],
                        y=[coords_df.iloc[example_node]['y'], coords_df.iloc[neighbor]['y']],
                        mode='lines',
                        line=dict(color='orange', width=2),
                        showlegend=False
                    ))

            # Highlight selected spot in red
            fig.add_trace(go.Scatter(
                x=[coords_df.iloc[example_node]['x']],
                y=[coords_df.iloc[example_node]['y']],
                mode='markers',
                marker=dict(color='red', size=15, symbol='star'),
                name='Selected Spot'
            ))

            fig.update_layout(
                title=f"Local Neighborhood of Spot {example_node}",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if track == "🔢 Mathematics-Focused Track":
            # Spectral analysis
            try:
                # Compute Laplacian eigenvalues (for small graphs)
                if len(G.nodes()) <= 500:  # Limit for computational efficiency
                    L = nx.normalized_laplacian_matrix(G).toarray()
                    eigenvals = np.linalg.eigvals(L)
                    eigenvals = np.sort(eigenvals)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Spectral Properties:**")
                        st.metric("Smallest Eigenvalue", f"{eigenvals[0]:.6f}")
                        st.metric("Largest Eigenvalue", f"{eigenvals[-1]:.3f}")
                        st.metric("Spectral Gap", f"{eigenvals[1] - eigenvals[0]:.3f}")

                        st.markdown("""
                        **Interpretation:**
                        - λ₁ = 0 (connected graph)
                        - λ₂ (Fiedler value): connectivity strength
                        - λₙ ≤ 2 (normalized Laplacian property)
                        """)

                    with col2:
                        fig = px.line(
                            x=range(len(eigenvals[:50])), y=eigenvals[:50],
                            title="Laplacian Eigenvalue Spectrum",
                            labels={'x': 'Index', 'y': 'Eigenvalue'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Spectral analysis skipped for large graphs (computational efficiency)")
            except Exception as e:
                st.warning("Spectral analysis not available (numerical issues)")
        else:
            st.markdown("""
            <div class="biological-insight">
            <h4>🧬 Why Graph Structure Matters</h4>

            The graph we've constructed encodes crucial biological principles:

            • **Local Communication**: Cells primarily influence nearby neighbors
            • **Spatial Constraints**: Physical barriers limit interaction
            • **Tissue Architecture**: Graph topology reflects underlying biology
            • **Signal Propagation**: Information flows through connected pathways

            This graph structure will guide how the neural network processes and learns from spatial gene expression patterns.
            </div>
            """, unsafe_allow_html=True)

# Section: GNN Architectures
elif section == "🧠 GNN Architectures":
    st.markdown('<div class="section-header">Graph Neural Network Architectures</div>', unsafe_allow_html=True)

    if track == "🧬 Biology-Focused Track":
        st.markdown("""
        <div class="biological-insight">
        <h4>🧠 Teaching Computers to Think Like Biologists</h4>

        <strong>The Challenge:</strong> How do we design neural networks that understand spatial tissue organization?

        <strong>The Solution:</strong> Graph Neural Networks that mimic how cells actually communicate:

        • **Message Passing**: Like chemical signals between neighboring cells
        • **Aggregation**: How cells integrate multiple signals from their environment  
        • **Transformation**: How cells process information to make decisions
        • **Layer-wise Processing**: Building up from local to global tissue understanding

        Different GNN architectures use different "communication strategies" to solve this problem.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="math-insight">
        <h4>🔢 Mathematical Framework of Graph Neural Networks</h4>

        <strong>General GNN Message Passing Framework:</strong>

        m_{ij}^{(l)} = M^{(l)}(h_i^{(l)}, h_j^{(l)}, e_{ij})  (Message function)

        m_i^{(l)} = AGG^{(l)}({m_{ij}^{(l)} : j ∈ N(i)})  (Aggregation function)

        h_i^{(l+1)} = U^{(l)}(h_i^{(l)}, m_i^{(l)})  (Update function)

        <strong>Key Design Choices:</strong>
        • **Message function M**: How to compute messages between nodes
        • **Aggregation AGG**: How to combine messages (sum, mean, max, attention)
        • **Update function U**: How to integrate aggregated messages with node state

        Different architectures make different choices for these functions.
        </div>
        """, unsafe_allow_html=True)

    # Architecture selector
    st.subheader("🏗️ Interactive Architecture Exploration")

    architecture = st.selectbox(
        "Select GNN Architecture to Explore:",
        ["Graph Convolutional Network (GCN)", "Graph Attention Network (GAT)",
         "GraphSAGE", "Graph Transformer"]
    )

    # Architecture-specific explanations
    if architecture == "Graph Convolutional Network (GCN)":
        col1, col2 = st.columns([1, 1])

        with col1:
            if track == "🧬 Biology-Focused Track":
                st.markdown("""
                <div class="biological-insight">
                <h4>🔬 GCN: Democratic Cell Communication</h4>

                <strong>Biological Analogy:</strong> Like a town hall meeting where every neighbor gets equal voice.

                <strong>How it works:</strong>
                1. Each cell listens to ALL its neighbors equally
                2. Averages all the signals it receives
                3. Combines this with its own state
                4. Updates its "opinion" (feature representation)

                <strong>Strengths:</strong>
                • Simple and effective
                • Works well when all neighbors are equally important
                • Fast computation

                <strong>Limitations:</strong>
                • Can't distinguish between different types of neighbors
                • May over-smooth distinct cell populations
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="math-insight">
                <h4>🔢 GCN Mathematical Formulation</h4>

                <strong>Layer-wise Propagation Rule:</strong>

                H^{(l+1)} = σ(D̃^{-1/2}ÃD̃^{-1/2}H^{(l)}W^{(l)})

                Where:
                • Ã = A + I (add self-loops)
                • D̃_{ii} = Σ_j Ã_{ij} (degree matrix)
                • W^{(l)} ∈ ℝ^{d_l × d_{l+1}} (learnable parameters)
                • σ: activation function (ReLU)

                <strong>Key Insight:</strong> Symmetric normalization D̃^{-1/2}ÃD̃^{-1/2} ensures:
                • Each node receives equal total message weight
                • Prevents high-degree nodes from dominating
                • Maintains numerical stability

                <strong>Computational Complexity:</strong> O(|E| · d · H) per layer
                </div>
                """, unsafe_allow_html=True)

        with col2:
            # GCN architecture diagram
            st.markdown("**GCN Layer Operation:**")

            fig = go.Figure()

            # Example neighborhood
            center = [0, 0]
            neighbors = [[-1, 1], [1, 1], [-1, -1], [1, -1]]

            # Draw edges
            for neighbor in neighbors:
                fig.add_trace(go.Scatter(
                    x=[center[0], neighbor[0]], y=[center[1], neighbor[1]],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False
                ))

            # Draw nodes
            fig.add_trace(go.Scatter(
                x=[center[0]], y=[center[1]],
                mode='markers+text',
                marker=dict(size=30, color='red'),
                text=['Central Cell'],
                textposition="bottom center",
                name='Target Node'
            ))

            neighbor_x, neighbor_y = zip(*neighbors)
            fig.add_trace(go.Scatter(
                x=neighbor_x, y=neighbor_y,
                mode='markers+text',
                marker=dict(size=20, color='blue'),
                text=['N1', 'N2', 'N3', 'N4'],
                textposition="top center",
                name='Neighbors'
            ))

            # Add aggregation arrow
            fig.add_annotation(
                x=0, y=-0.7,
                text="Equal Weight<br>Aggregation",
                showarrow=False,
                font=dict(size=12)
            )

            fig.update_layout(
                title="GCN: Equal Neighbor Weighting",
                height=300,
                xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False)
            )

            st.plotly_chart(fig, use_container_width=True)

            if track == "🔢 Mathematics-Focused Track":
                st.code("""
# GCN Layer Implementation
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A_norm):
        # X: [n, in_features], A_norm: [n, n]
        # Aggregate neighbors then transform
        X_agg = torch.mm(A_norm, X)  # Aggregate
        X_out = self.linear(X_agg)   # Transform
        return F.relu(X_out)         # Activate
                """, language='python')

    elif architecture == "Graph Attention Network (GAT)":
        col1, col2 = st.columns([1, 1])

        with col1:
            if track == "🧬 Biology-Focused Track":
                st.markdown("""
                <div class="biological-insight">
                <h4>🔬 GAT: Selective Cell Communication</h4>

                <strong>Biological Analogy:</strong> Like a smart cell that pays more attention to important neighbors.

                <strong>How it works:</strong>
                1. Cell evaluates each neighbor's relevance
                2. Assigns attention weights based on compatibility
                3. Focuses more on important signals, less on noise
                4. Different cell types can attend to different neighbors

                <strong>Biological Relevance:</strong>
                • Cancer cells might attend more to growth signals
                • Immune cells focus on inflammatory signals
                • Normal cells ignore aberrant cancer signals

                <strong>Advantages:</strong>
                • Learns which neighbors matter most
                • Interpretable attention weights
                • Better handles heterogeneous neighborhoods
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="math-insight">
                <h4>🔢 GAT Attention Mechanism</h4>

                <strong>Attention Coefficient Computation:</strong>

                e_{ij} = a(W·h_i, W·h_j)  (Attention energy)

                α_{ij} = softmax_j(e_{ij}) = exp(e_{ij})/Σ_{k∈N_i} exp(e_{ik})

                <strong>Message Aggregation:</strong>

                h_i' = σ(Σ_{j∈N_i} α_{ij} W h_j)

                <strong>Multi-Head Attention:</strong>

                h_i' = ||_{k=1}^K σ(Σ_{j∈N_i} α_{ij}^k W^k h_j)

                Where || denotes concatenation

                <strong>Computational Complexity:</strong> O(|E| · d · H + |V| · H²)

                <strong>Learnable Parameters:</strong> W ∈ ℝ^{d×d'}, a ∈ ℝ^{2d'}
                </div>
                """, unsafe_allow_html=True)

        with col2:
            # GAT attention visualization
            st.markdown("**GAT Attention Mechanism:**")

            fig = go.Figure()

            # Example with different attention weights
            center = [0, 0]
            neighbors = [[-1, 1], [1, 1], [-1, -1], [1, -1]]
            attention_weights = [0.1, 0.5, 0.3, 0.1]  # Different importance

            # Draw edges with varying thickness based on attention
            for neighbor, weight in zip(neighbors, attention_weights):
                fig.add_trace(go.Scatter(
                    x=[center[0], neighbor[0]], y=[center[1], neighbor[1]],
                    mode='lines',
                    line=dict(color='orange', width=weight * 10),
                    showlegend=False
                ))

                # Add attention weight labels
                mid_x, mid_y = (center[0] + neighbor[0]) / 2, (center[1] + neighbor[1]) / 2
                fig.add_annotation(
                    x=mid_x, y=mid_y,
                    text=f"α={weight:.1f}",
                    showarrow=False,
                    font=dict(size=10, color='red')
                )

            # Draw nodes
            fig.add_trace(go.Scatter(
                x=[center[0]], y=[center[1]],
                mode='markers+text',
                marker=dict(size=30, color='red'),
                text=['Central Cell'],
                textposition="bottom center",
                name='Target Node'
            ))

            neighbor_x, neighbor_y = zip(*neighbors)
            fig.add_trace(go.Scatter(
                x=neighbor_x, y=neighbor_y,
                mode='markers+text',
                marker=dict(size=20, color='blue'),
                text=['N1', 'N2', 'N3', 'N4'],
                textposition="top center",
                name='Neighbors'
            ))

            fig.update_layout(
                title="GAT: Learned Attention Weights",
                height=300,
                xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False)
            )

            st.plotly_chart(fig, use_container_width=True)

            if track == "🔢 Mathematics-Focused Track":
                st.code("""
# GAT Attention Layer
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=1):
        super().__init__()
        self.W = nn.Linear(in_features, out_features * heads)
        self.a = nn.Parameter(torch.randn(2 * out_features, 1))
        self.heads = heads

    def forward(self, X, edge_index):
        # Transform features
        H = self.W(X).view(-1, self.heads, self.out_features)

        # Compute attention coefficients
        edge_h = torch.cat([H[edge_index[0]], H[edge_index[1]]], dim=-1)
        e = torch.matmul(edge_h, self.a).squeeze()
        alpha = softmax(e, edge_index[0])

        # Aggregate with attention
        out = scatter_add(alpha.unsqueeze(-1) * H[edge_index[1]], 
                         edge_index[0], dim=0)
        return out.view(-1, self.heads * self.out_features)
                """, language='python')

    elif architecture == "GraphSAGE":
        col1, col2 = st.columns([1, 1])

        with col1:
            if track == "🧬 Biology-Focused Track":
                st.markdown("""
                <div class="biological-insight">
                <h4>🔬 GraphSAGE: Inductive Cell Learning</h4>

                <strong>Biological Analogy:</strong> Like learning general rules about cell behavior that apply to new tissue samples.

                <strong>Key Innovation:</strong>
                1. Learns to aggregate neighborhood information generally
                2. Can handle new cells/tissues not seen during training
                3. Samples neighborhoods to scale to large tissues

                <strong>Aggregation Strategies:</strong>
                • **Mean**: Average of neighbor features
                • **LSTM**: Sequential processing of neighbors  
                • **Pool**: Max/mean pooling of transformed neighbors

                <strong>Clinical Relevance:</strong>
                • Train on one patient's tumor, apply to another
                • Handle different tissue sizes and architectures
                • Scale to whole-slide imaging data
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="math-insight">
                <h4>🔢 GraphSAGE Inductive Framework</h4>

                <strong>Sampling-based Aggregation:</strong>

                h_{N(v)}^{(l)} = AGGREGATE_l({h_u^{(l-1)} : u ∈ S(N(v))})

                h_v^{(l)} = σ(W^{(l)} · CONCAT(h_v^{(l-1)}, h_{N(v)}^{(l)}))

                Where S(N(v)) is a sampled subset of neighbors

                <strong>Aggregation Functions:</strong>

                • **Mean**: AGGREGATE = 1/|S| Σ_{u∈S} h_u^{(l-1)}
                • **LSTM**: AGGREGATE = LSTM({h_u^{(l-1)} : u ∈ S})  
                • **Pool**: AGGREGATE = max({σ(W_pool h_u^{(l-1)}) : u ∈ S})

                <strong>Inductive Property:**
                Parameters are independent of graph structure → generalize to unseen graphs

                <strong>Sampling Complexity:** O(|S|^L) for L layers, |S| samples per layer
                </div>
                """, unsafe_allow_html=True)

        with col2:
            # GraphSAGE sampling visualization
            st.markdown("**GraphSAGE Neighborhood Sampling:**")

            # Create a multi-layer sampling visualization
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=["Layer 0 (Input)", "Layer 1 Sample", "Layer 2 Sample"],
                specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]]
            )

            # Layer 0: Full neighborhood
            center = [0, 0]
            layer0_neighbors = [[-1, 1], [1, 1], [-1, -1], [1, -1], [0, 1.5], [0, -1.5]]

            for neighbor in layer0_neighbors:
                fig.add_trace(go.Scatter(
                    x=[center[0], neighbor[0]], y=[center[1], neighbor[1]],
                    mode='lines', line=dict(color='lightgray', width=1),
                    showlegend=False
                ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=[center[0]], y=[center[1]],
                mode='markers', marker=dict(size=15, color='red'),
                showlegend=False
            ), row=1, col=1)

            neighbor_x, neighbor_y = zip(*layer0_neighbors)
            fig.add_trace(go.Scatter(
                x=neighbor_x, y=neighbor_y,
                mode='markers', marker=dict(size=10, color='lightblue'),
                showlegend=False
            ), row=1, col=1)

            # Layer 1: Sampled neighborhood
            sampled_neighbors = [[-1, 1], [1, 1], [-1, -1]]  # Sample 3 out of 6

            for neighbor in sampled_neighbors:
                fig.add_trace(go.Scatter(
                    x=[center[0], neighbor[0]], y=[center[1], neighbor[1]],
                    mode='lines', line=dict(color='orange', width=2),
                    showlegend=False
                ), row=1, col=2)

            fig.add_trace(go.Scatter(
                x=[center[0]], y=[center[1]],
                mode='markers', marker=dict(size=15, color='red'),
                showlegend=False
            ), row=1, col=2)

            sampled_x, sampled_y = zip(*sampled_neighbors)
            fig.add_trace(go.Scatter(
                x=sampled_x, y=sampled_y,
                mode='markers', marker=dict(size=10, color='blue'),
                showlegend=False
            ), row=1, col=2)

            # Layer 2: Further sampling
            final_neighbors = [[-1, 1], [1, 1]]  # Sample 2 out of 3

            for neighbor in final_neighbors:
                fig.add_trace(go.Scatter(
                    x=[center[0], neighbor[0]], y=[center[1], neighbor[1]],
                    mode='lines', line=dict(color='green', width=3),
                    showlegend=False
                ), row=1, col=3)

            fig.add_trace(go.Scatter(
                x=[center[0]], y=[center[1]],
                mode='markers', marker=dict(size=15, color='red'),
                showlegend=False
            ), row=1, col=3)

            final_x, final_y = zip(*final_neighbors)
            fig.add_trace(go.Scatter(
                x=final_x, y=final_y,
                mode='markers', marker=dict(size=10, color='darkgreen'),
                showlegend=False
            ), row=1, col=3)

            fig.update_layout(height=250, showlegend=False)
            for i in range(1, 4):
                fig.update_xaxes(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False, row=1, col=i)
                fig.update_yaxes(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False, row=1, col=i)

            st.plotly_chart(fig, use_container_width=True)

            if track == "🔢 Mathematics-Focused Track":
                st.code("""
# GraphSAGE with Mean Aggregation
class SAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.self_linear = nn.Linear(in_features, out_features)
        self.neigh_linear = nn.Linear(in_features, out_features)

    def forward(self, X, edge_index, sample_size=10):
        # Sample neighborhoods
        sampled_edges = sample_neighbors(edge_index, sample_size)

        # Aggregate neighbor features (mean)
        neigh_agg = scatter_mean(X[sampled_edges[1]], 
                                sampled_edges[0], dim=0)

        # Combine self and neighbor information
        self_out = self.self_linear(X)
        neigh_out = self.neigh_linear(neigh_agg)

        return F.relu(self_out + neigh_out)
                """, language='python')

    elif architecture == "Graph Transformer":
        col1, col2 = st.columns([1, 1])

        with col1:
            if track == "🧬 Biology-Focused Track":
                st.markdown("""
                <div class="biological-insight">
                <h4>🔬 Graph Transformer: Global Cell Communication</h4>

                <strong>Biological Analogy:</strong> Like cells that can sense and respond to signals from across the entire tissue, not just immediate neighbors.

                <strong>How it works:</strong>
                1. Each cell can attend to ANY other cell in the tissue
                2. Learns which distant cells are relevant for each decision
                3. Captures long-range dependencies in tissue architecture
                4. Uses positional encoding to maintain spatial awareness

                <strong>Biological Applications:</strong>
                • Metastatic spread patterns (long-range)
                • Hormone signaling across tissue regions
                • Immune surveillance networks
                • Vascular/neural connectivity patterns

                <strong>Trade-offs:**
                • More expressive but computationally expensive
                • Can capture global patterns missed by local methods
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="math-insight">
                <h4>🔢 Graph Transformer Architecture</h4>

                <strong>Self-Attention on Graphs:</strong>

                Q, K, V = XW_Q, XW_K, XW_V

                A_{ij} = softmax((Q_i K_j^T)/√d + b_{ij})

                Where b_{ij} encodes structural/positional information

                <strong>Output Computation:</strong>

                O_i = Σ_j A_{ij} V_j

                <strong>Positional Encoding:**
                • Laplacian PE: Use eigenvectors of graph Laplacian
                • Distance PE: Encode shortest path distances
                • Spatial PE: Use actual coordinates (x, y)

                <strong>Complexity:** O(n² · d) for full attention

                <strong>Sparse Attention:** Limit to k-hop neighborhoods → O(|E| · d)
                </div>
                """, unsafe_allow_html=True)

        with col2:
            # Graph Transformer attention pattern
            st.markdown("**Graph Transformer Global Attention:**")

            fig = go.Figure()

            # Create a grid of nodes
            positions = []
            for i in range(5):
                for j in range(5):
                    positions.append([i - 2, j - 2])

            # Show attention from center node to all others
            center_idx = 12  # Middle of 5x5 grid
            attention_weights = np.random.exponential(0.3, 25)
            attention_weights[center_idx] = 1.0  # Self-attention
            attention_weights = attention_weights / attention_weights.sum()

            # Draw connections with varying opacity based on attention
            for i, (pos, weight) in enumerate(zip(positions, attention_weights)):
                if i != center_idx:
                    opacity = min(weight * 20, 1.0)  # Scale for visibility
                    fig.add_trace(go.Scatter(
                        x=[positions[center_idx][0], pos[0]],
                        y=[positions[center_idx][1], pos[1]],
                        mode='lines',
                        line=dict(color='purple', width=1),
                        opacity=opacity,
                        showlegend=False
                    ))

            # Draw all nodes
            x_coords, y_coords = zip(*positions)
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers',
                marker=dict(
                    size=8,
                    color=['red' if i == center_idx else 'lightblue' for i in range(25)],
                    line=dict(width=1, color='white')
                ),
                showlegend=False
            ))

            # Highlight center node
            fig.add_trace(go.Scatter(
                x=[positions[center_idx][0]], y=[positions[center_idx][1]],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name='Query Node'
            ))

            fig.update_layout(
                title="Global Attention Pattern",
                height=300,
                xaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[-2.5, 2.5], showgrid=False, zeroline=False, showticklabels=False)
            )

            st.plotly_chart(fig, use_container_width=True)

            if track == "🔢 Mathematics-Focused Track":
                st.code("""
# Graph Transformer Layer
class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )

    def forward(self, X, pos_encoding, attention_mask=None):
        # Add positional encoding
        X_pos = X + pos_encoding

        # Self-attention
        attn_out, _ = self.attention(X_pos, X_pos, X_pos, 
                                   attn_mask=attention_mask)
        X = self.norm1(X + attn_out)

        # Feed-forward
        ffn_out = self.ffn(X)
        X = self.norm2(X + ffn_out)

        return X
                """, language='python')

    # Implementation comparison
    st.subheader("💻 Architecture Implementation")

    with st.expander(f"View {architecture} Implementation Details"):
        if track == "🧬 Biology-Focused Track":
            st.markdown(f"""
            **Implementing {architecture} for Spatial Transcriptomics:**

            1. **Data Preprocessing:**
               - Load spatial coordinates and gene expression
               - Normalize expression data (log-transform, standardize)
               - Construct spatial graph based on proximity

            2. **Model Architecture:**
               - Input layer: Gene expression features
               - {architecture} layers: Learn spatial patterns
               - Output layer: Embeddings or predictions

            3. **Training Process:**
               - Define loss function (reconstruction, classification, etc.)
               - Use backpropagation to learn spatial relationships
               - Validate on held-out tissue regions

            4. **Biological Interpretation:**
               - Analyze learned embeddings for cell types
               - Identify spatially variable genes
               - Discover tissue architectural patterns
            """)
        else:
            # Show full implementation for selected architecture
            if architecture == "Graph Convolutional Network (GCN)":
                st.code("""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SpatialGCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(SpatialGCN, self).__init__()

        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims)-1):
            self.layers.append(GCNConv(dims[i], dims[i+1]))

        self.dropout = dropout

    def forward(self, x, edge_index):
        layer_outputs = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            layer_outputs.append(x)

            # Apply activation and dropout (except last layer)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return x, layer_outputs

# Usage example
model = SpatialGCN(
    input_dim=50,      # Number of genes
    hidden_dims=[32, 16], # Hidden layer sizes
    output_dim=8,      # Final embedding dimension
    dropout=0.1
)

# Forward pass
embeddings, all_layers = model(gene_expression, edge_index)
                """, language='python')

            elif architecture == "Graph Attention Network (GAT)":
                st.code("""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SpatialGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 heads=4, dropout=0.1):
        super(SpatialGAT, self).__init__()

        self.conv1 = GATConv(input_dim, hidden_dim, 
                           heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim,
                           heads=heads, dropout=dropout)  
        self.conv3 = GATConv(hidden_dim * heads, output_dim,
                           heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, return_attention_weights=False):
        # Layer 1
        x, attn1 = self.conv1(x, edge_index, 
                            return_attention_weights=True)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Layer 2  
        x, attn2 = self.conv2(x, edge_index,
                            return_attention_weights=True)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Layer 3
        x, attn3 = self.conv3(x, edge_index,
                            return_attention_weights=True)

        if return_attention_weights:
            return x, (attn1, attn2, attn3)
        return x

# Attention visualization
model = SpatialGAT(input_dim=50, hidden_dim=32, output_dim=8)
embeddings, attention_weights = model(gene_expression, edge_index, 
                                    return_attention_weights=True)
                """, language='python')

# Section: Architecture Comparison
elif section == "⚖️ Architecture Comparison":
    st.markdown('<div class="section-header">Comprehensive Graph Neural Network Comparison</div>',
                unsafe_allow_html=True)

    if track == "🧬 Biology-Focused Track":
        st.markdown("""
        <div class="biological-insight">
        <h4>🔬 Choosing the Right GNN for Your Biological Question</h4>

        Different graph neural networks excel at different types of biological problems. 
        Think of them as different "microscopes" - each reveals different aspects of spatial biology.

        The choice depends on:
        • **Tissue heterogeneity**: How diverse are the cell types?
        • **Spatial patterns**: Local vs. global relationships?
        • **Data scale**: Single tissue section vs. whole-slide imaging?
        • **Interpretability needs**: Do you need to understand the "why"?
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="math-insight">
        <h4>🔢 Comparative Analysis Framework</h4>

        <strong>Evaluation Dimensions:</strong>
        • **Expressiveness**: What patterns can the model capture?
        • **Computational Complexity**: Time and space requirements
        • **Inductive Capability**: Generalization to new graphs
        • **Interpretability**: Understanding learned representations
        • **Scalability**: Performance on large graphs

        <strong>Mathematical Properties:**
        • **Permutation Invariance**: f(πX) = πf(X)
        • **Locality**: k-layer GNN sees k-hop neighborhoods
        • **Over-smoothing**: Deep networks may lose distinctiveness
        • **Expressive Power**: Theoretical limitations (1-WL test)
        </div>
        """, unsafe_allow_html=True)

    # Comparison table
    st.subheader("📊 Architecture Comparison Matrix")

    comparison_data = {
        'Architecture': ['GCN', 'GAT', 'GraphSAGE', 'Graph Transformer'],
        'Computational Complexity': ['O(|E|·d)', 'O(|E|·d + |V|·H²)', 'O(|S|^L·d)', 'O(|V|²·d)'],
        'Memory Requirements': ['Low', 'Medium', 'Medium', 'High'],
        'Interpretability': ['Low', 'High', 'Medium', 'High'],
        'Scalability': ['Excellent', 'Good', 'Excellent', 'Limited'],
        'Inductive Learning': ['Limited', 'Limited', 'Excellent', 'Good'],
        'Best For': ['Homogeneous tissues', 'Heterogeneous regions', 'Large-scale data', 'Global patterns']
    }

    if track == "🧬 Biology-Focused Track":
        # Biological use cases
        comparison_data['Biological Use Case'] = [
            'Uniform cell populations',
            'Tumor-immune interfaces',
            'Multi-patient studies',
            'Metastasis tracking'
        ]
        comparison_data['Clinical Application'] = [
            'Biomarker discovery',
            'Drug target identification',
            'Diagnostic tools',
            'Treatment planning'
        ]
    else:
        # Mathematical properties
        comparison_data['Theoretical Foundation'] = [
            'Spectral graph theory',
            'Attention mechanism',
            'Sampling theory',
            'Transformer architecture'
        ]
        comparison_data['Approximation Quality'] = [
            'First-order spectral',
            'Learned attention',
            'Sampling-based',
            'Full attention'
        ]

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # Detailed comparison sections
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Performance Comparison", "Use Case Analysis", "Implementation Trade-offs", "Future Directions"])

    with tab1:
        st.subheader("🎯 Performance on Spatial Transcriptomics Tasks")

        # Simulate performance comparison
        np.random.seed(42)
        architectures = ['GCN', 'GAT', 'GraphSAGE', 'Graph Transformer']
        tasks = ['Cell Type Classification', 'Spatial Domain Detection', 'Gene Expression Prediction',
                 'Biomarker Discovery']

        # Generate simulated performance scores
        performance_data = []
        for arch in architectures:
            for task in tasks:
                # Simulate realistic performance differences
                base_score = np.random.normal(0.75, 0.05)
                if arch == 'GAT' and 'Classification' in task:
                    base_score += 0.1  # GAT better at classification
                elif arch == 'GraphSAGE' and 'Prediction' in task:
                    base_score += 0.08  # GraphSAGE better at prediction
                elif arch == 'Graph Transformer' and 'Biomarker' in task:
                    base_score += 0.12  # Transformer better at global patterns

                performance_data.append({
                    'Architecture': arch,
                    'Task': task,
                    'Performance': min(max(base_score, 0.5), 0.95)  # Clamp to realistic range
                })

        perf_df = pd.DataFrame(performance_data)

        fig = px.bar(
            perf_df, x='Task', y='Performance', color='Architecture',
            title="Performance Comparison Across Tasks",
            barmode='group'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        if track == "🧬 Biology-Focused Track":
            st.markdown("""
            <div class="biological-insight">
            <strong>Key Insights:</strong>

            • **GCN**: Best baseline for most tasks, especially when all neighbors are equally important
            • **GAT**: Excels when different cell types have different interaction patterns
            • **GraphSAGE**: Superior for studies involving multiple patients or tissue types
            • **Graph Transformer**: Best for discovering long-range spatial patterns

            <strong>Recommendation:</strong> Start with GCN for proof-of-concept, then upgrade based on specific biological needs.
            </div>
            """, unsafe_allow_html=True)
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **Statistical Significance Testing:**

                For robust comparison, consider:
                • Multiple random seeds (≥5)
                • Cross-validation (k-fold)
                • Statistical tests (paired t-test)
                • Effect size (Cohen's d)
                • Confidence intervals
                """)

            with col2:
                st.markdown("""
                **Hyperparameter Sensitivity:**

                Architecture-specific considerations:
                • GCN: Layer depth, dropout rate
                • GAT: Number of attention heads, attention dropout  
                • GraphSAGE: Sampling size, aggregation function
                • Transformer: Positional encoding, attention span
                """)

    with tab2:
        st.subheader("🔬 Use Case Decision Tree")

        if track == "🧬 Biology-Focused Track":
            # Interactive decision tree for biologists
            st.markdown("**Answer these questions to find the best architecture for your research:**")

            tissue_type = st.selectbox(
                "What type of tissue are you studying?",
                ["Tumor (heterogeneous)", "Normal tissue (homogeneous)", "Immune-rich regions", "Mixed/unknown"]
            )

            analysis_goal = st.selectbox(
                "What's your primary analysis goal?",
                ["Discover new cell types", "Predict treatment response", "Find spatial biomarkers",
                 "Understand cell communication"]
            )

            data_scale = st.selectbox(
                "What's your data scale?",
                ["Single tissue section", "Multiple patients", "Whole-slide imaging", "Time series"]
            )

            interpretability = st.selectbox(
                "How important is interpretability?",
                ["Critical (need to explain to clinicians)", "Important (for publication)",
                 "Moderate (performance first)", "Not important"]
            )

            # Decision logic
            if st.button("Get Architecture Recommendation"):
                recommendation = "GCN"  # Default
                reasoning = []

                if tissue_type == "Tumor (heterogeneous)":
                    if interpretability in ["Critical (need to explain to clinicians)", "Important (for publication)"]:
                        recommendation = "GAT"
                        reasoning.append("GAT provides interpretable attention weights for heterogeneous tumor regions")
                    else:
                        recommendation = "Graph Transformer"
                        reasoning.append("Graph Transformer captures complex heterogeneous patterns")

                elif analysis_goal == "Understand cell communication":
                    if data_scale == "Whole-slide imaging":
                        recommendation = "Graph Transformer"
                        reasoning.append("Graph Transformer captures long-range communication patterns")
                    else:
                        recommendation = "GAT"
                        reasoning.append("GAT attention weights reveal communication patterns")

                elif data_scale in ["Multiple patients", "Time series"]:
                    recommendation = "GraphSAGE"
                    reasoning.append("GraphSAGE excels at inductive learning across different samples")

                elif analysis_goal == "Find spatial biomarkers":
                    if tissue_type == "Immune-rich regions":
                        recommendation = "GAT"
                        reasoning.append("GAT can identify which immune cells contribute to biomarker patterns")
                    else:
                        recommendation = "Graph Transformer"
                        reasoning.append("Graph Transformer finds global biomarker patterns")

                elif tissue_type == "Normal tissue (homogeneous)":
                    recommendation = "GCN"
                    reasoning.append("GCN is efficient and effective for homogeneous tissue patterns")

                # Display recommendation
                st.success(f"**Recommended Architecture: {recommendation}**")

                for reason in reasoning:
                    st.write(f"• {reason}")

                # Additional considerations
                st.markdown(f"""
                <div class="biological-insight">
                <h4>Additional Considerations for {recommendation}:</h4>

                {'<strong>Implementation tips:</strong><br>• Start with 2-3 layers<br>• Use attention dropout (0.1-0.2)<br>• Visualize attention weights for validation' if recommendation == 'GAT' else ''}
                {'<strong>Implementation tips:</strong><br>• Use neighborhood sampling (10-25 neighbors)<br>• Consider mean aggregation for stability<br>• Test on multiple tissue types' if recommendation == 'GraphSAGE' else ''}
                {'<strong>Implementation tips:</strong><br>• Add positional encoding for spatial awareness<br>• Use sparse attention for large graphs<br>• Monitor computational requirements' if recommendation == 'Graph Transformer' else ''}
                {'<strong>Implementation tips:</strong><br>• Start simple with 2-3 layers<br>• Add self-loops for stability<br>• Use dropout for regularization' if recommendation == 'GCN' else ''}
                </div>
                """, unsafe_allow_html=True)

        else:  # Math track
            st.markdown("""
            <div class="math-insight">
            <h4>🔢 Mathematical Decision Criteria</h4>

            <strong>Graph Properties → Architecture Choice:</strong>

            **Homophily Analysis:**
            • High homophily (similar nodes connect): GCN works well
            • Low homophily (dissimilar nodes connect): GAT or GraphSAGE

            **Graph Size and Density:**
            • Small graphs (|V| < 1000): Any architecture
            • Medium graphs (1000 < |V| < 10000): GCN, GAT, GraphSAGE
            • Large graphs (|V| > 10000): GraphSAGE, sparse attention

            **Spectral Properties:**
            • Large spectral gap: GCN sufficient
            • Small spectral gap: Need more expressive models

            **Computational Budget:**
            • O(|E|): GCN
            • O(|E| + |V|h²): GAT
            • O(|V|²): Graph Transformer (full attention)
            </div>
            """, unsafe_allow_html=True)

            # Mathematical analysis tool
            st.markdown("**Graph Property Calculator:**")

            col1, col2 = st.columns(2)

            with col1:
                # Calculate actual graph properties
                edges, _ = create_spatial_graph(coords_df[['x', 'y']].values)
                G = nx.Graph()
                G.add_edges_from(edges.T)

                # Homophily calculation
                region_labels = coords_df['region'].values
                homophily_edges = 0
                total_edges = 0

                for edge in edges.T:
                    i, j = edge[0], edge[1]
                    if region_labels[i] == region_labels[j]:
                        homophily_edges += 1
                    total_edges += 1

                homophily = homophily_edges / total_edges if total_edges > 0 else 0

                st.metric("Graph Homophily", f"{homophily:.3f}")
                st.metric("Average Clustering", f"{nx.average_clustering(G):.3f}")
                st.metric("Graph Density", f"{nx.density(G):.4f}")

                # Recommendation based on properties
                if homophily > 0.7:
                    st.success("High homophily → GCN recommended")
                elif homophily < 0.3:
                    st.warning("Low homophily → GAT/GraphSAGE recommended")
                else:
                    st.info("Medium homophily → Any architecture suitable")

            with col2:
                # Computational complexity comparison
                n_nodes = len(coords_df)
                n_edges = len(edges.T)
                d_features = 50
                h_heads = 4

                complexity_data = {
                    'Architecture': ['GCN', 'GAT', 'GraphSAGE', 'Graph Transformer'],
                    'Time Complexity': [
                        f"O({n_edges} × {d_features}) = {n_edges * d_features:,}",
                        f"O({n_edges} × {d_features} + {n_nodes} × {h_heads}²) = {n_edges * d_features + n_nodes * h_heads ** 2:,}",
                        f"O(S^L × {d_features}) ≈ {(10 ** 2) * d_features:,}",
                        f"O({n_nodes}² × {d_features}) = {n_nodes ** 2 * d_features:,}"
                    ],
                    'Relative Cost': [1, 1.2, 0.4, n_nodes / 10]  # Approximate relative costs
                }

                complexity_df = pd.DataFrame(complexity_data)
                st.dataframe(complexity_df, use_container_width=True)

    with tab3:
        st.subheader("⚙️ Implementation Trade-offs")

        if track == "🧬 Biology-Focused Track":
            st.markdown("""
            <div class="biological-insight">
            <h4>🔬 Practical Considerations for Biologists</h4>

            <strong>Getting Started (Easiest to Hardest):</strong>
            1. **GCN**: Simple, well-documented, many tutorials
            2. **GraphSAGE**: Good documentation, inductive learning
            3. **GAT**: Attention visualization requires extra work
            4. **Graph Transformer**: Complex, cutting-edge, fewer resources

            <strong>Data Requirements:</strong>
            • **GCN**: Works with any spatial transcriptomics data
            • **GAT**: Benefits from heterogeneous cell populations
            • **GraphSAGE**: Needs multiple samples for full potential
            • **Graph Transformer**: Requires large datasets for training
            </div>
            """, unsafe_allow_html=True)

            # Implementation difficulty comparison
            difficulty_data = {
                'Architecture': ['GCN', 'GAT', 'GraphSAGE', 'Graph Transformer'],
                'Implementation Difficulty': [2, 4, 3, 5],
                'Documentation Quality': [5, 4, 4, 3],
                'Community Support': [5, 4, 4, 2],
                'Debugging Difficulty': [2, 4, 3, 5]
            }

            difficulty_df = pd.DataFrame(difficulty_data)

            fig = px.radar(
                difficulty_df, r='Implementation Difficulty', theta='Architecture',
                title="Implementation Difficulty (1=Easy, 5=Hard)"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:  # Math track
            st.markdown("""
            <div class="math-insight">
            <h4>🔢 Implementation Complexity Analysis</h4>

            <strong>Memory Requirements:</strong>

            **GCN:**
            • Parameters: O(d₁d₂L) where L = layers
            • Activations: O(|V|d)
            • Sparse adjacency: O(|E|)

            **GAT:**
            • Parameters: O(d₁d₂L + d₂H) where H = heads
            • Attention weights: O(|E|H)
            • Additional memory for multi-head computation

            **GraphSAGE:**
            • Parameters: O(d₁d₂L)
            • Sampling buffer: O(S^L × d)
            • Reduced memory through sampling

            **Graph Transformer:**
            • Parameters: O(d²L + d⁴) for FFN
            • Attention matrix: O(|V|²) (full) or O(|E|) (sparse)
            • Position encodings: O(|V|d)
            </div>
            """, unsafe_allow_html=True)

            # Memory usage visualization
            n_nodes = len(coords_df)
            n_edges = len(edges.T)
            d_model = 64
            n_layers = 3
            n_heads = 4

            memory_estimates = {
                'Component': ['Parameters', 'Activations', 'Graph Structure', 'Architecture-Specific'],
                'GCN': [
                    d_model ** 2 * n_layers,
                    n_nodes * d_model,
                    n_edges,
                    0
                ],
                'GAT': [
                    d_model ** 2 * n_layers + d_model * n_heads,
                    n_nodes * d_model,
                    n_edges,
                    n_edges * n_heads  # Attention weights
                ],
                'GraphSAGE': [
                    d_model ** 2 * n_layers,
                    (10 ** n_layers) * d_model,  # Sampling buffer
                    n_edges,
                    0
                ],
                'Graph Transformer': [
                    d_model ** 2 * n_layers + d_model ** 2 * 4,  # FFN
                    n_nodes * d_model,
                    n_edges,
                    n_nodes * d_model  # Position encoding
                ]
            }

            memory_df = pd.DataFrame(memory_estimates)

            fig = px.bar(
                memory_df.melt(id_vars=['Component'], var_name='Architecture', value_name='Memory'),
                x='Architecture', y='Memory', color='Component',
                title="Memory Requirements Breakdown (Relative Units)",
                log_y=True
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🚀 Future Directions and Emerging Architectures")

        if track == "🧬 Biology-Focused Track":
            st.markdown("""
            <div class="biological-insight">
            <h4>🔬 Next-Generation Spatial Biology Methods</h4>

            <strong>Emerging Trends:</strong>

            **1. Multi-Modal Integration:**
            • Combine transcriptomics + histology + proteomics
            • Graph neural networks for multi-omic spatial data
            • Applications: Comprehensive tumor characterization

            **2. Temporal Spatial Analysis:**
            • Track how spatial patterns change over time
            • Treatment response monitoring
            • Disease progression modeling

            **3. 3D Spatial Reconstruction:**
            • Move beyond 2D tissue sections
            • True 3D tissue architecture analysis
            • Organ-level spatial patterns

            **4. Real-time Clinical Applications:**
            • Intraoperative spatial analysis
            • Rapid diagnostic tools
            • Personalized treatment selection
            </div>
            """, unsafe_allow_html=True)

            # Timeline of developments
            timeline_data = {
                'Year': [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
                'Development': [
                    'GCN introduced',
                    'GAT published',
                    'GraphSAGE for inductive learning',
                    'First spatial transcriptomics + GNN papers',
                    'Graph Transformers emerge',
                    'Multi-modal spatial analysis',
                    'Clinical validation studies',
                    'Regulatory approval approaches',
                    'Routine clinical implementation'
                ],
                'Impact': [3, 4, 4, 5, 3, 4, 3, 2, 5]
            }

            timeline_df = pd.DataFrame(timeline_data)

            fig = px.line(
                timeline_df, x='Year', y='Impact',
                title="Evolution of Spatial GNN Methods",
                hover_data=['Development']
            )
            fig.update_traces(mode='markers+lines', marker_size=8)
            st.plotly_chart(fig, use_container_width=True)

        else:  # Math track
            st.markdown("""
            <div class="math-insight">
            <h4>🔢 Theoretical Advances and Open Problems</h4>

            <strong>Current Limitations:</strong>

            **1. Expressiveness Bounds:**
            • Most GNNs limited by 1-Weisfeiler-Lehman test
            • Cannot distinguish certain graph structures
            • Need for higher-order methods

            **2. Over-smoothing Problem:**
            • Deep GNNs lose node distinctiveness
            • Mathematical analysis: eigenvalue perspective
            • Solutions: residual connections, normalization

            **3. Heterophily Challenge:**
            • Standard GNNs assume homophily
            • Real biological networks often heterophilic
            • Need for specialized architectures

            **4. Scalability Issues:**
            • Quadratic complexity in transformers
            • Memory limitations for large graphs
            • Distributed training challenges
            </div>
            """, unsafe_allow_html=True)

            # Research directions matrix
            research_data = {
                'Problem': ['Expressiveness', 'Over-smoothing', 'Heterophily', 'Scalability', 'Interpretability'],
                'Current Solutions': [
                    'Higher-order GNNs, Graph transformers',
                    'Residual connections, Normalization',
                    'Signed attention, Heterophilic GNNs',
                    'Sampling, Sparse attention',
                    'Attention visualization, GNNExplainer'
                ],
                'Open Questions': [
                    'Optimal expressiveness-efficiency trade-off',
                    'Theoretical depth limits',
                    'Universal heterophily handling',
                    'Linear-time global attention',
                    'Causal vs. correlational explanations'
                ],
                'Mathematical Complexity': ['High', 'Medium', 'High', 'Medium', 'High']
            }

            research_df = pd.DataFrame(research_data)
            st.dataframe(research_df, use_container_width=True)

# Section: Training & Analysis
elif section == "🔬 Training & Analysis":
    st.markdown('<div class="section-header">GNN Training and Analysis Pipeline</div>', unsafe_allow_html=True)

    # Prepare data for analysis
    edges, weights = create_spatial_graph(coords_df[['x', 'y']].values)
    scaler = StandardScaler()
    expression_scaled = scaler.fit_transform(expression_df.values)

    # Convert to PyTorch tensors
    x = torch.FloatTensor(expression_scaled)
    edge_index = torch.LongTensor(edges)

    # Model selection
    architecture_choice = st.selectbox(
        "Select architecture for training demo:",
        ["GCN", "GAT", "GraphSAGE", "Graph Transformer"]
    )

    # Initialize model based on choice
    if architecture_choice == "GCN":
        model = SpatialGCN(input_dim=50, hidden_dim=32, output_dim=8)
    elif architecture_choice == "GAT":
        model = SpatialGAT(input_dim=50, hidden_dim=32, output_dim=8, heads=4)
    elif architecture_choice == "GraphSAGE":
        model = SpatialSAGE(input_dim=50, hidden_dim=32, output_dim=8)
    else:  # Graph Transformer
        model = SpatialTransformer(input_dim=50, hidden_dim=32, output_dim=8, heads=4)

    model.eval()

    with torch.no_grad():
        embeddings, layer2, layer1 = model(x, edge_index)

    # Convert back to numpy
    embeddings_np = embeddings.numpy()
    layer1_np = layer1.numpy()
    layer2_np = layer2.numpy()

    if track == "🧬 Biology-Focused Track":
        st.markdown(f"""
        <div class="biological-insight">
        <h4>🔬 Understanding {architecture_choice} Training</h4>

        <strong>What the model learns:</strong>
        • **Layer 1**: Basic spatial neighborhoods (who's next to whom)
        • **Layer 2**: Multi-hop patterns (neighborhoods of neighborhoods)
        • **Layer 3**: Complex spatial signatures unique to each tissue region

        <strong>Training process:</strong>
        1. Model sees spatial graph + gene expression
        2. Learns to predict tissue regions from spatial patterns
        3. Discovers which gene combinations are spatially important
        4. Creates embeddings that separate different tissue types
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="math-insight">
        <h4>🔢 {architecture_choice} Training Mathematics</h4>

        <strong>Objective Function:</strong>
        L = L_supervised + λL_regularization

        **Supervised Loss** (classification): L_sup = -Σᵢ yᵢ log(softmax(f(xᵢ)))

        **Regularization**: L_reg = ||θ||₂² (L2 penalty)

        <strong>Optimization:</strong>
        • Algorithm: Adam with learning rate schedule
        • Batch size: Full graph (transductive)
        • Gradient flow: Through message passing operations

        <strong>Convergence Properties:</strong>
        • Non-convex optimization landscape
        • Local minima depend on initialization
        • Early stopping prevents overfitting
        </div>
        """, unsafe_allow_html=True)

    # Training analysis
    st.subheader("📊 Training Results Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Embeddings Visualization", "Layer Analysis", "Performance Metrics", "Interpretability"])

    with tab1:
        # PCA visualization of embeddings
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_np)

        col1, col2 = st.columns(2)

        with col1:
            # Original spatial layout
            fig = px.scatter(
                coords_df, x='x', y='y', color='region_name',
                title="Original Tissue Regions",
                color_discrete_map={
                    'Tumor Core': '#d62728',
                    'Tumor Edge': '#ff7f0e',
                    'Immune Zone': '#2ca02c',
                    'Normal Tissue': '#1f77b4'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Learned embeddings
            plot_df = coords_df.copy()
            plot_df['PC1'] = embeddings_2d[:, 0]
            plot_df['PC2'] = embeddings_2d[:, 1]

            fig = px.scatter(
                plot_df, x='PC1', y='PC2', color='region_name',
                title=f"{architecture_choice} Learned Embeddings (PCA)",
                color_discrete_map={
                    'Tumor Core': '#d62728',
                    'Tumor Edge': '#ff7f0e',
                    'Immune Zone': '#2ca02c',
                    'Normal Tissue': '#1f77b4'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        if track == "🧬 Biology-Focused Track":
            st.markdown("""
            <div class="biological-insight">
            <strong>Key Observation:</strong> Notice how the learned embeddings (right) show clearer separation between tissue regions compared to the spatial layout (left). 

            The model has learned to map cells with similar spatial contexts to similar points in embedding space, even if they're physically distant.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Quantitative separation analysis
            from sklearn.metrics import silhouette_score, adjusted_rand_score
            from sklearn.cluster import KMeans

            # Clustering on embeddings
            kmeans = KMeans(n_clusters=4, random_state=42)
            predicted_clusters = kmeans.fit_predict(embeddings_np)

            # Metrics
            silhouette = silhouette_score(embeddings_np, coords_df['region'])
            ari = adjusted_rand_score(coords_df['region'], predicted_clusters)

            explained_var = pca.explained_variance_ratio_

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score", f"{silhouette:.3f}")
            with col2:
                st.metric("Adjusted Rand Index", f"{ari:.3f}")
            with col3:
                st.metric("PC1+PC2 Variance Explained", f"{sum(explained_var[:2]):.1%}")

    with tab2:
        # Layer-by-layer analysis
        st.subheader("🔍 Layer-wise Feature Evolution")

        layer_data = {
            "Input (Standardized)": expression_scaled,
            "Layer 1 Output": layer1_np,
            "Layer 2 Output": layer2_np,
            "Final Embeddings": embeddings_np
        }

        selected_layer = st.selectbox("Select layer to analyze:", list(layer_data.keys()))
        current_data = layer_data[selected_layer]

        # PCA for visualization
        if current_data.shape[1] > 2:
            pca_layer = PCA(n_components=2)
            layer_2d = pca_layer.fit_transform(current_data)
            explained_var = pca_layer.explained_variance_ratio_
        else:
            layer_2d = current_data
            explained_var = [1.0, 0.0]

        col1, col2 = st.columns(2)

        with col1:
            # Spatial layout colored by first PC
            plot_df = coords_df.copy()
            plot_df['PC1'] = layer_2d[:, 0]

            fig = px.scatter(
                plot_df, x='x', y='y', color='PC1',
                title=f"Spatial Pattern - {selected_layer}",
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Feature space plot
            plot_df = coords_df.copy()
            plot_df['PC1'] = layer_2d[:, 0]
            plot_df['PC2'] = layer_2d[:, 1]

            fig = px.scatter(
                plot_df, x='PC1', y='PC2', color='region_name',
                title=f"Feature Space - {selected_layer}",
                color_discrete_map={
                    'Tumor Core': '#d62728',
                    'Tumor Edge': '#ff7f0e',
                    'Immune Zone': '#2ca02c',
                    'Normal Tissue': '#1f77b4'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        if track == "🔢 Mathematics-Focused Track":
            st.markdown(f"""
            **Layer Statistics:**
            - Feature dimension: {current_data.shape[1]}
            - PC1 variance explained: {explained_var[0]:.1%}
            - PC2 variance explained: {explained_var[1]:.1%}
            - Total variance (PC1+PC2): {sum(explained_var[:2]):.1%}

            **Feature Evolution:** As we go deeper, representations become more task-specific and less interpretable in terms of original genes.
            """)

    with tab3:
        # Performance metrics
        st.subheader("📈 Model Performance Analysis")

        if track == "🧬 Biology-Focused Track":
            # Clustering evaluation
            from sklearn.cluster import KMeans
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

            n_clusters = st.slider("Number of clusters for evaluation:", 2, 8, 4)

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings_np)

            # Calculate metrics
            ari = adjusted_rand_score(coords_df['region'], clusters)
            nmi = normalized_mutual_info_score(coords_df['region'], clusters)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Adjusted Rand Index", f"{ari:.3f}")
                st.caption("Measures cluster-region alignment")

            with col2:
                st.metric("Normalized Mutual Info", f"{nmi:.3f}")
                st.caption("Information shared between clusters and regions")

            with col3:
                # Homogeneity of clusters
                cluster_purity = []
                for cluster_id in range(n_clusters):
                    cluster_mask = clusters == cluster_id
                    if cluster_mask.sum() > 0:
                        cluster_regions = coords_df.loc[cluster_mask, 'region']
                        most_common = cluster_regions.mode().iloc[0] if len(cluster_regions.mode()) > 0 else 0
                        purity = (cluster_regions == most_common).sum() / len(cluster_regions)
                        cluster_purity.append(purity)

                avg_purity = np.mean(cluster_purity) if cluster_purity else 0
                st.metric("Average Cluster Purity", f"{avg_purity:.3f}")
                st.caption("How homogeneous are the clusters")

            # Visualize clustering results
            plot_df = coords_df.copy()
            plot_df['predicted_cluster'] = clusters.astype(str)
            plot_df['true_region'] = plot_df['region_name']

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Predicted Clusters", "True Regions"],
                specs=[[{"type": "scatter"}, {"type": "scatter"}]]
            )

            # Predicted clusters
            for cluster_id in range(n_clusters):
                mask = clusters == cluster_id
                fig.add_trace(
                    go.Scatter(
                        x=coords_df.loc[mask, 'x'],
                        y=coords_df.loc[mask, 'y'],
                        mode='markers',
                        marker=dict(size=8),
                        name=f'Cluster {cluster_id}',
                        showlegend=False
                    ),
                    row=1, col=1
                )

            # True regions
            region_colors = {'Tumor Core': 'red', 'Tumor Edge': 'orange', 'Immune Zone': 'green',
                             'Normal Tissue': 'blue'}
            for region_name, color in region_colors.items():
                mask = coords_df['region_name'] == region_name
                fig.add_trace(
                    go.Scatter(
                        x=coords_df.loc[mask, 'x'],
                        y=coords_df.loc[mask, 'y'],
                        mode='markers',
                        marker=dict(size=8, color=color),
                        name=region_name,
                        showlegend=False
                    ),
                    row=1, col=2
                )

            fig.update_layout(height=400, title=f"Clustering Performance (ARI: {ari:.3f})")
            st.plotly_chart(fig, use_container_width=True)

        else:  # Math track
            # Mathematical performance analysis
            st.markdown("""
            **Quantitative Evaluation Metrics:**

            For unsupervised spatial analysis, we evaluate:
            """)

            # Calculate various metrics
            from scipy.spatial.distance import pdist, squareform
            from sklearn.metrics import pairwise_distances


            # Trustworthiness and continuity
            def trustworthiness_continuity(X_high, X_low, k=10):
                n = X_high.shape[0]

                # Distance matrices
                D_high = pairwise_distances(X_high)
                D_low = pairwise_distances(X_low)

                # k-nearest neighbors in both spaces
                nn_high = np.argsort(D_high, axis=1)[:, 1:k + 1]
                nn_low = np.argsort(D_low, axis=1)[:, 1:k + 1]

                # Trustworthiness
                trustworthiness = 0
                for i in range(n):
                    for j in nn_low[i]:
                        if j not in nn_high[i]:
                            rank_high = np.where(np.argsort(D_high[i]) == j)[0][0]
                            trustworthiness += max(0, rank_high - k)

                trustworthiness = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * trustworthiness

                # Continuity
                continuity = 0
                for i in range(n):
                    for j in nn_high[i]:
                        if j not in nn_low[i]:
                            rank_low = np.where(np.argsort(D_low[i]) == j)[0][0]
                            continuity += max(0, rank_low - k)

                continuity = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * continuity

                return trustworthiness, continuity


            # Calculate metrics
            trust, cont = trustworthiness_continuity(expression_scaled, embeddings_np, k=10)


            # Neighborhood preservation
            def neighborhood_preservation(X_high, X_low, k=10):
                n = X_high.shape[0]
                D_high = pairwise_distances(X_high)
                D_low = pairwise_distances(X_low)

                nn_high = np.argsort(D_high, axis=1)[:, 1:k + 1]
                nn_low = np.argsort(D_low, axis=1)[:, 1:k + 1]

                preservation = 0
                for i in range(n):
                    intersection = len(set(nn_high[i]) & set(nn_low[i]))
                    preservation += intersection / k

                return preservation / n


            neighborhood_pres = neighborhood_preservation(expression_scaled, embeddings_np, k=10)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Trustworthiness", f"{trust:.3f}")
                st.caption("Avoids false neighbors in embedding")

            with col2:
                st.metric("Continuity", f"{cont:.3f}")
                st.caption("Preserves true neighbors")

            with col3:
                st.metric("Neighborhood Preservation", f"{neighborhood_pres:.3f}")
                st.caption("10-NN overlap between spaces")

            # Stress and other metrics
            st.markdown("""
            **Interpretation:**
            - **Trustworthiness > 0.8**: Embedding faithfully represents local structure
            - **Continuity > 0.8**: Few true neighbors are lost in embedding
            - **Neighborhood Preservation > 0.6**: Good local structure preservation
            """)

    with tab4:
        # Interpretability analysis
        st.subheader("🔍 Model Interpretability")

        if track == "🧬 Biology-Focused Track":
            st.markdown("""
            <div class="biological-insight">
            <h4>🔬 Understanding What the Model Learned</h4>

            <strong>Key Questions:</strong>
            • Which genes are most important for spatial patterns?
            • How do spatial neighborhoods influence gene expression?
            • Can we identify novel spatial biomarkers?
            • What tissue architecture features does the model recognize?
            </div>
            """, unsafe_allow_html=True)

            # Gene importance analysis
            st.markdown("**Gene Importance Analysis:**")

            embedding_dim = st.selectbox("Select embedding dimension:", range(8))

            # Calculate correlation between input genes and selected embedding dimension
            gene_importance = []
            for i, gene in enumerate(expression_df.columns):
                corr = np.corrcoef(expression_df[gene], embeddings_np[:, embedding_dim])[0, 1]
                gene_importance.append({
                    'Gene': gene,
                    'Correlation': corr,
                    'Absolute_Correlation': abs(corr)
                })

            importance_df = pd.DataFrame(gene_importance)
            importance_df = importance_df.sort_values('Absolute_Correlation', ascending=False)

            # Top genes
            top_genes = importance_df.head(15)

            fig = px.bar(
                top_genes, x='Correlation', y='Gene',
                title=f"Gene Contributions to Embedding Dimension {embedding_dim}",
                orientation='h',
                color='Correlation',
                color_continuous_scale='RdBu'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Biological interpretation
            top_gene = top_genes.iloc[0]['Gene']
            top_corr = top_genes.iloc[0]['Correlation']

            st.markdown(f"""
            <div class="biological-insight">
            <strong>Interpretation:</strong> {top_gene} shows the strongest correlation ({top_corr:.3f}) with embedding dimension {embedding_dim}.

            {'This suggests the model learned that ' + top_gene + ' expression patterns are crucial for distinguishing spatial regions.' if abs(top_corr) > 0.5 else 'The model uses multiple genes together rather than relying on single markers.'}
            </div>
            """, unsafe_allow_html=True)

        else:  # Math track
            st.markdown("""
            <div class="math-insight">
            <h4>🔢 Mathematical Interpretability Methods</h4>

            <strong>Gradient-based Importance:</strong>
            ∂L/∂x_i measures how much loss changes with input gene i

            <strong>Attention Weights (for GAT):</strong>
            α_ij = attention from node j to node i

            <strong>Feature Ablation:</strong>
            Compare performance with/without specific genes

            <strong>Graph Connectivity Analysis:</strong>
            Which edges are most important for predictions?
            </div>
            """, unsafe_allow_html=True)

            # Mathematical analysis
            if architecture_choice == "GAT":
                st.markdown("**Attention Pattern Analysis (GAT-specific):**")

                # Simulate attention weights (in real implementation, extract from model)
                np.random.seed(42)
                attention_stats = []

                for region in coords_df['region'].unique():
                    region_mask = coords_df['region'] == region
                    region_name = coords_df[region_mask]['region_name'].iloc[0]

                    # Simulate attention diversity for this region
                    attention_entropy = np.random.uniform(1.0, 3.0)  # Higher = more diverse attention
                    avg_attention = np.random.uniform(0.1, 0.9)

                    attention_stats.append({
                        'Region': region_name,
                        'Attention_Entropy': attention_entropy,
                        'Average_Attention': avg_attention,
                        'Attention_Diversity': attention_entropy / np.log(6)  # Normalize by max possible entropy
                    })

                attention_df = pd.DataFrame(attention_stats)

                fig = px.bar(
                    attention_df, x='Region', y='Attention_Diversity',
                    title="Attention Diversity by Tissue Region",
                    color='Attention_Diversity',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                **Interpretation:**
                - **High diversity**: Region has heterogeneous neighborhoods requiring selective attention
                - **Low diversity**: Region has homogeneous neighborhoods with uniform attention
                """)

            else:
                # General interpretability for other architectures
                st.markdown("**Embedding Space Analysis:**")

                # PCA loadings analysis
                pca = PCA(n_components=3)
                pca.fit(embeddings_np)

                # Analyze which original features contribute to PC1
                # For this, we need to trace back through the network (simplified analysis)

                loadings_data = []
                for i, gene in enumerate(expression_df.columns[:20]):  # Subset for display
                    # Simplified: correlation between input gene and first PC
                    pc1_scores = pca.transform(embeddings_np)[:, 0]
                    correlation = np.corrcoef(expression_scaled[:, i], pc1_scores)[0, 1]

                    loadings_data.append({
                        'Gene': gene,
                        'PC1_Loading': correlation,
                        'Contribution': abs(correlation)
                    })

                loadings_df = pd.DataFrame(loadings_data)
                loadings_df = loadings_df.sort_values('Contribution', ascending=False)

                fig = px.scatter(
                    loadings_df, x='PC1_Loading', y='Gene',
                    size='Contribution',
                    title="Gene Contributions to Principal Component 1",
                    hover_data=['Contribution']
                )
                st.plotly_chart(fig, use_container_width=True)

# Section: Applications & Insights
elif section == "💡 Applications & Insights":
    st.markdown('<div class="section-header">From Analysis to Biological Discovery</div>', unsafe_allow_html=True)

    if track == "🧬 Biology-Focused Track":
        st.markdown("""
        <div class="biological-insight">
        <h4>🎯 Translating Computational Results to Biological Impact</h4>

        <strong>The Ultimate Goal:</strong> Use spatial transcriptomics + GNNs to answer fundamental questions in cancer biology and develop better treatments.

        <strong>Key Applications:</strong>
        • **Drug Target Discovery**: Find genes expressed specifically in tumor regions
        • **Biomarker Development**: Identify spatial signatures that predict treatment response
        • **Personalized Medicine**: Tailor treatments based on individual tumor architecture
        • **Clinical Decision Support**: Guide surgical and therapeutic interventions
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="math-insight">
        <h4>🔢 From Mathematical Models to Clinical Applications</h4>

        <strong>Translation Pipeline:</strong>

        1. **Model Development**: Optimize architectures for spatial data
        2. **Validation**: Prove performance on benchmark datasets  
        3. **Clinical Testing**: Validate on patient samples
        4. **Regulatory Approval**: FDA/EMA pathway for diagnostic tools
        5. **Clinical Implementation**: Integration into healthcare workflows

        <strong>Success Metrics:</strong>
        • Sensitivity/Specificity for diagnostic applications
        • Hazard ratios for prognostic biomarkers
        • Area under ROC curve for predictive models
        </div>
        """, unsafe_allow_html=True)

    # Generate embeddings for analysis
    edges, weights = create_spatial_graph(coords_df[['x', 'y']].values)
    scaler = StandardScaler()
    expression_scaled = scaler.fit_transform(expression_df.values)

    x = torch.FloatTensor(expression_scaled)
    edge_index = torch.LongTensor(edges)

    model = SpatialGCN(input_dim=50, hidden_dim=32, output_dim=8)
    model.eval()

    with torch.no_grad():
        embeddings, _, _ = model(x, edge_index)

    embeddings_np = embeddings.numpy()

    # Application areas
    st.subheader("🔬 Practical Applications")

    if track == "🧬 Biology-Focused Track":
        app_tabs = st.tabs([
            "🎯 Drug Target Discovery",
            "🔬 Biomarker Development",
            "💊 Treatment Prediction",
            "🏥 Clinical Translation"
        ])
    else:
        app_tabs = st.tabs([
            "🎯 Optimization Problems",
            "📊 Statistical Analysis",
            "🤖 Machine Learning Pipeline",
            "📈 Performance Validation"
        ])

    with app_tabs[0]:
        if track == "🧬 Biology-Focused Track":
            st.markdown("### Drug Target Discovery")

            st.markdown("""
            <div class="biological-insight">
            <h4>🎯 Finding Spatial-Specific Drug Targets</h4>

            <strong>Research Question:</strong> Which genes are specifically upregulated in tumor regions and could serve as therapeutic targets?

            <strong>Advantage of Spatial Analysis:</strong> Traditional bulk RNA-seq might miss tumor-specific targets because they average expression across all cell types. Spatial analysis preserves this crucial information.
            </div>
            """, unsafe_allow_html=True)

            # Identify tumor-specific targets
            tumor_mask = coords_df['region'].isin([0, 1])  # Tumor core and edge
            normal_mask = coords_df['region'].isin([2, 3])  # Immune and normal

            drug_targets = []
            for gene in expression_df.columns[:25]:  # Analyze subset
                tumor_expr = expression_df.loc[tumor_mask, gene].mean()
                normal_expr = expression_df.loc[normal_mask, gene].mean()

                # Calculate tumor specificity
                specificity = tumor_expr / (normal_expr + 1e-6)
                expression_level = tumor_expr

                # Good targets: high in tumor, low in normal
                if specificity > 2.0 and expression_level > 1.0:
                    drug_targets.append({
                        'Gene': gene,
                        'Tumor_Expression': tumor_expr,
                        'Normal_Expression': normal_expr,
                        'Specificity_Ratio': specificity,
                        'Druggability_Score': specificity * expression_level
                    })

            if drug_targets:
                targets_df = pd.DataFrame(drug_targets)
                targets_df = targets_df.sort_values('Druggability_Score', ascending=False)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Top Drug Target Candidates:**")
                    display_df = targets_df[['Gene', 'Specificity_Ratio', 'Druggability_Score']].head(10)
                    st.dataframe(display_df.round(2), use_container_width=True)

                    if len(targets_df) > 0:
                        st.success(f"Identified {len(targets_df)} potential targets with tumor specificity!")

                with col2:
                    fig = px.scatter(
                        targets_df, x='Normal_Expression', y='Tumor_Expression',
                        size='Druggability_Score', hover_name='Gene',
                        title="Drug Target Landscape",
                        labels={'x': 'Normal Tissue Expression', 'y': 'Tumor Expression'}
                    )
                    # Add diagonal line (equal expression)
                    max_expr = max(targets_df['Normal_Expression'].max(), targets_df['Tumor_Expression'].max())
                    fig.add_shape(
                        type="line", x0=0, y0=0, x1=max_expr, y1=max_expr,
                        line=dict(color="red", dash="dash")
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Spatial visualization of top target
                if len(targets_df) > 0:
                    top_target = targets_df.iloc[0]['Gene']

                    plot_df = coords_df.copy()
                    plot_df['target_expression'] = expression_df[top_target]

                    fig = px.scatter(
                        plot_df, x='x', y='y', color='target_expression',
                        title=f"Spatial Expression: {top_target} (Top Drug Target)",
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(f"""
                    <div class="biological-insight">
                    <strong>Clinical Potential:</strong> {top_target} shows strong tumor-specific expression patterns. 
                    This spatial specificity suggests it could be a good therapeutic target with minimal off-target effects in normal tissue.

                    <strong>Next Steps:</strong>
                    • Validate in additional patient samples
                    • Check for existing inhibitors/antibodies
                    • Test therapeutic efficacy in preclinical models
                    </div>
                    """, unsafe_allow_html=True)

        else:  # Math track
            st.markdown("### Optimization for Target Discovery")

            st.markdown("""
            <div class="math-insight">
            <h4>🔢 Mathematical Optimization Framework</h4>

            <strong>Multi-objective Optimization Problem:</strong>

            maximize: f(g) = w₁ · specificity(g) + w₂ · expression(g) - w₃ · toxicity(g)

            subject to:
            • specificity(g) = E[X_g | tumor] / E[X_g | normal] ≥ τ₁
            • expression(g) = E[X_g | tumor] ≥ τ₂  
            • druggability(g) ∈ {druggable set}

            <strong>Pareto Optimal Solutions:</strong>
            Cannot improve one objective without worsening another
            </div>
            """, unsafe_allow_html=True)

            # Mathematical analysis
            tumor_mask = coords_df['region'].isin([0, 1])
            normal_mask = coords_df['region'].isin([2, 3])

            # Calculate metrics for all genes
            optimization_results = []
            for gene in expression_df.columns[:30]:
                tumor_expr = expression_df.loc[tumor_mask, gene].mean()
                normal_expr = expression_df.loc[normal_mask, gene].mean()
                tumor_std = expression_df.loc[tumor_mask, gene].std()

                specificity = tumor_expr / (normal_expr + 1e-6)
                expression_level = tumor_expr
                variability = tumor_std / (tumor_expr + 1e-6)  # Coefficient of variation

                # Multi-objective score
                w1, w2, w3 = 0.4, 0.4, 0.2  # Weights
                score = w1 * np.log(specificity) + w2 * np.log(expression_level + 1) - w3 * variability

                optimization_results.append({
                    'Gene': gene,
                    'Specificity': specificity,
                    'Expression': expression_level,
                    'Variability': variability,
                    'Optimization_Score': score
                })

            opt_df = pd.DataFrame(optimization_results)
            opt_df = opt_df.sort_values('Optimization_Score', ascending=False)

            # Pareto frontier analysis
            fig = px.scatter(
                opt_df, x='Specificity', y='Expression',
                size='Optimization_Score', hover_name='Gene',
                color='Variability',
                title="Multi-objective Optimization Space",
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Mathematical metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Pareto Optimal Solutions",
                          len(opt_df[opt_df['Optimization_Score'] > opt_df['Optimization_Score'].quantile(0.8)]))
            with col2:
                st.metric("Specificity Range", f"{opt_df['Specificity'].min():.2f} - {opt_df['Specificity'].max():.2f}")
            with col3:
                st.metric("Expression Range", f"{opt_df['Expression'].min():.2f} - {opt_df['Expression'].max():.2f}")

    with app_tabs[1]:
        if track == "🧬 Biology-Focused Track":
            st.markdown("### Spatial Biomarker Development")

            st.markdown("""
            <div class="biological-insight">
            <h4>🔬 Discovering Spatial Biomarkers</h4>

            <strong>Goal:</strong> Identify spatial gene expression patterns that predict patient outcomes or treatment responses.

            <strong>Why Spatial Matters:</strong> 
            • Tumor architecture affects drug penetration
            • Immune infiltration patterns predict immunotherapy response
            • Spatial heterogeneity indicates treatment resistance potential
            </div>
            """, unsafe_allow_html=True)

            # Boundary detection for biomarker discovery
            from sklearn.neighbors import NearestNeighbors
            from scipy.spatial.distance import pdist

            # Find spots with high embedding diversity in neighborhood (boundaries)
            nbrs = NearestNeighbors(n_neighbors=6).fit(coords_df[['x', 'y']].values)
            _, indices = nbrs.kneighbors(coords_df[['x', 'y']].values)

            boundary_scores = []
            for i in range(len(coords_df)):
                neighbor_embeddings = embeddings_np[indices[i]]
                diversity = np.std(pdist(neighbor_embeddings))
                boundary_scores.append(diversity)

            boundary_scores = np.array(boundary_scores)
            is_boundary = boundary_scores > np.percentile(boundary_scores, 75)

            col1, col2 = st.columns(2)

            with col1:
                # Boundary detection results
                st.markdown("**Spatial Boundary Detection:**")
                st.metric("Boundary Spots Detected", is_boundary.sum())
                st.metric("Core Spots", (~is_boundary).sum())
                st.metric("Boundary Percentage", f"{100 * is_boundary.sum() / len(coords_df):.1f}%")

                # Find boundary-enriched genes
                boundary_genes = []
                for gene in expression_df.columns[:20]:
                    boundary_expr = expression_df.loc[is_boundary, gene].mean()
                    core_expr = expression_df.loc[~is_boundary, gene].mean()
                    fold_change = boundary_expr / (core_expr + 1e-6)

                    if fold_change > 1.5:  # Upregulated at boundaries
                        boundary_genes.append({
                            'Gene': gene,
                            'Fold_Change': fold_change,
                            'P_value': 0.001 * np.random.random()  # Simulated
                        })

                if boundary_genes:
                    boundary_df = pd.DataFrame(boundary_genes)
                    st.markdown("**Boundary-Enriched Biomarkers:**")
                    st.dataframe(boundary_df.round(3), use_container_width=True)

            with col2:
                # Visualize boundary detection
                plot_df = coords_df.copy()
                plot_df['boundary_score'] = boundary_scores
                plot_df['is_boundary'] = is_boundary

                fig = px.scatter(
                    plot_df, x='x', y='y',
                    color='boundary_score',
                    title="Spatial Boundary Detection",
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Clinical application
            st.markdown("""
            <div class="biological-insight">
            <h4>🏥 Clinical Application of Spatial Biomarkers</h4>

            <strong>Boundary-enriched genes could predict:</strong>
            • Tumor invasiveness (genes active at invasion front)
            • Treatment resistance (stress response at boundaries)
            • Metastatic potential (EMT markers at tumor edge)

            <strong>Clinical workflow:</strong>
            1. Patient biopsy → Spatial transcriptomics
            2. GNN analysis → Boundary detection
            3. Biomarker scoring → Risk stratification
            4. Treatment selection → Personalized therapy
            </div>
            """, unsafe_allow_html=True)

        else:  # Math track
            st.markdown("### Statistical Validation Framework")

            st.markdown("""
            <div class="math-insight">
            <h4>📊 Biomarker Statistical Testing</h4>

            <strong>Hypothesis Testing Framework:</strong>

            H₀: μ_boundary = μ_core (no difference in expression)
            H₁: μ_boundary ≠ μ_core (significant spatial difference)

            <strong>Test Statistics:</strong>
            • t-test: assumes normal distribution
            • Mann-Whitney U: non-parametric alternative
            • Permutation test: spatial-aware null distribution

            <strong>Multiple Testing Correction:</strong>
            • Bonferroni: α' = α/m (conservative)
            • FDR (Benjamini-Hochberg): controls false discovery rate
            • Spatial FDR: accounts for spatial correlation
            </div>
            """, unsafe_allow_html=True)

            # Statistical analysis
            from scipy import stats

            # Boundary detection (same as biology track)
            nbrs = NearestNeighbors(n_neighbors=6).fit(coords_df[['x', 'y']].values)
            _, indices = nbrs.kneighbors(coords_df[['x', 'y']].values)

            boundary_scores = []
            for i in range(len(coords_df)):
                neighbor_embeddings = embeddings_np[indices[i]]
                diversity = np.std(pdist(neighbor_embeddings))
                boundary_scores.append(diversity)

            boundary_scores = np.array(boundary_scores)
            is_boundary = boundary_scores > np.percentile(boundary_scores, 75)

            # Statistical testing for each gene
            statistical_results = []
            for gene in expression_df.columns[:25]:
                boundary_expr = expression_df.loc[is_boundary, gene]
                core_expr = expression_df.loc[~is_boundary, gene]

                # t-test
                t_stat, t_pval = stats.ttest_ind(boundary_expr, core_expr)

                # Mann-Whitney U test
                u_stat, u_pval = stats.mannwhitneyu(boundary_expr, core_expr, alternative='two-sided')

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(boundary_expr) - 1) * boundary_expr.var() +
                                      (len(core_expr) - 1) * core_expr.var()) /
                                     (len(boundary_expr) + len(core_expr) - 2))
                cohens_d = (boundary_expr.mean() - core_expr.mean()) / pooled_std

                statistical_results.append({
                    'Gene': gene,
                    'T_statistic': t_stat,
                    'T_pvalue': t_pval,
                    'U_pvalue': u_pval,
                    'Cohens_d': cohens_d,
                    'Effect_size': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
                })

            stats_df = pd.DataFrame(statistical_results)

            # FDR correction
            from statsmodels.stats.multitest import multipletests

            _, fdr_pvals, _, _ = multipletests(stats_df['T_pvalue'], method='fdr_bh')
            stats_df['FDR_pvalue'] = fdr_pvals

            # Significant results
            significant = stats_df[stats_df['FDR_pvalue'] < 0.05].sort_values('FDR_pvalue')

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Statistical Summary:**")
                st.metric("Genes Tested", len(stats_df))
                st.metric("Significant (FDR < 0.05)", len(significant))
                st.metric("Large Effect Size", len(stats_df[abs(stats_df['Cohens_d']) > 0.8]))

                if len(significant) > 0:
                    st.markdown("**Top Significant Genes:**")
                    display_cols = ['Gene', 'FDR_pvalue', 'Cohens_d', 'Effect_size']
                    st.dataframe(significant[display_cols].head(10).round(4), use_container_width=True)

            with col2:
                # Volcano plot
                stats_df['neg_log_pval'] = -np.log10(stats_df['FDR_pvalue'] + 1e-10)

                fig = px.scatter(
                    stats_df, x='Cohens_d', y='neg_log_pval',
                    hover_name='Gene',
                    title="Volcano Plot: Boundary vs Core Expression",
                    labels={'x': "Effect Size (Cohen's d)", 'y': '-log₁₀(FDR p-value)'}
                )

                # Add significance lines
                fig.add_hline(y=-np.log10(0.05), line_dash="dash", annotation_text="FDR = 0.05")
                fig.add_vline(x=0.5, line_dash="dot", annotation_text="Medium Effect")
                fig.add_vline(x=-0.5, line_dash="dot")

                st.plotly_chart(fig, use_container_width=True)

    with app_tabs[2]:
        if track == "🧬 Biology-Focused Track":
            st.markdown("### Treatment Response Prediction")

            st.markdown("""
            <div class="biological-insight">
            <h4>💊 Predicting Treatment Response from Spatial Patterns</h4>

            <strong>Clinical Challenge:</strong> Not all patients respond to the same treatment. Can spatial gene expression patterns predict who will benefit from specific therapies?

            <strong>Spatial Factors Affecting Treatment:</strong>
            • Drug penetration barriers (dense vs sparse tissue)
            • Immune infiltration (affects immunotherapy response)
            • Tumor heterogeneity (resistance cell populations)
            • Vascular architecture (affects drug delivery)
            </div>
            """, unsafe_allow_html=True)

            # Simulate treatment response prediction
            np.random.seed(42)

            # Create synthetic treatment response based on spatial features
            response_probability = np.zeros(len(coords_df))

            for i in range(len(coords_df)):
                # Factors affecting response
                region = coords_df.iloc[i]['region']

                # Immune infiltration (positive for immunotherapy)
                immune_factor = 0.8 if region == 2 else 0.2  # Higher response in immune zones

                # Tumor core (often treatment resistant)
                core_factor = 0.3 if region == 0 else 0.7  # Lower response in tumor core

                # Spatial diversity (heterogeneous regions harder to treat)
                neighbors = []
                for edge in edges.T:
                    if edge[0] == i:
                        neighbors.append(edge[1])
                    elif edge[1] == i:
                        neighbors.append(edge[0])

                if neighbors:
                    neighbor_regions = coords_df.iloc[neighbors]['region']
                    diversity = len(set(neighbor_regions)) / len(neighbor_regions)
                    diversity_factor = 1 - diversity  # Lower response with higher diversity
                else:
                    diversity_factor = 0.5

                # Combined probability
                prob = 0.3 * immune_factor + 0.4 * core_factor + 0.3 * diversity_factor
                response_probability[i] = prob

            # Add noise and create binary response
            response_probability += np.random.normal(0, 0.1, len(coords_df))
            response_probability = np.clip(response_probability, 0, 1)
            predicted_response = response_probability > 0.5

            col1, col2 = st.columns(2)

            with col1:
                # Response prediction results
                st.markdown("**Treatment Response Prediction:**")
                st.metric("Predicted Responders", predicted_response.sum())
                st.metric("Predicted Non-responders", (~predicted_response).sum())
                st.metric("Response Rate", f"{100 * predicted_response.sum() / len(coords_df):.1f}%")

                # Response by region
                response_by_region = []
                for region in coords_df['region'].unique():
                    region_mask = coords_df['region'] == region
                    region_name = coords_df[region_mask]['region_name'].iloc[0]
                    region_response_rate = predicted_response[region_mask].mean()

                    response_by_region.append({
                        'Region': region_name,
                        'Response_Rate': region_response_rate,
                        'Sample_Size': region_mask.sum()
                    })

                response_df = pd.DataFrame(response_by_region)

                fig = px.bar(
                    response_df, x='Region', y='Response_Rate',
                    title="Predicted Response Rate by Tissue Region",
                    color='Response_Rate',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Spatial map of response prediction
                plot_df = coords_df.copy()
                plot_df['response_probability'] = response_probability
                plot_df['predicted_response'] = predicted_response

                fig = px.scatter(
                    plot_df, x='x', y='y',
                    color='response_probability',
                    title="Spatial Treatment Response Prediction",
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0.5
                )
                st.plotly_chart(fig, use_container_width=True)

            # Clinical implementation
            st.markdown("""
            <div class="biological-insight">
            <h4>🏥 Clinical Implementation Strategy</h4>

            <strong>Personalized Treatment Selection:</strong>

            **High Response Probability (>70%):**
            • Standard first-line therapy
            • Monitor for early response markers

            **Medium Response Probability (30-70%):**
            • Combination therapy approaches
            • Enhanced monitoring and support

            **Low Response Probability (<30%):**
            • Alternative treatment strategies
            • Clinical trial consideration
            • Palliative care planning

            <strong>Validation Requirements:</strong>
            • Prospective clinical trials
            • Multi-center validation
            • Regulatory approval pathway
            </div>
            """, unsafe_allow_html=True)

        else:  # Math track
            st.markdown("### Machine Learning Pipeline")

            st.markdown("""
            <div class="math-insight">
            <h4>🤖 Predictive Modeling Framework</h4>

            <strong>Supervised Learning Setup:</strong>

            • Features: X ∈ ℝ^{n×d} (GNN embeddings)
            • Labels: y ∈ {0,1}^n (treatment response)
            • Objective: Learn f: X → y with high accuracy

            <strong>Model Architecture:</strong>

            GNN Encoder → Embeddings → MLP Classifier → Response Probability

            <strong>Loss Function:</strong>
            L = -∑ᵢ [yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)] + λ||θ||₂²

            Where p̂ᵢ = σ(W·embed(xᵢ) + b)
            </div>
            """, unsafe_allow_html=True)

            # Machine learning pipeline
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

            # Generate synthetic response labels
            np.random.seed(42)

            # Create response labels based on embeddings + noise
            # Simulate that first embedding dimension is predictive
            response_scores = embeddings_np[:, 0] + 0.1 * embeddings_np[:, 1] + np.random.normal(0, 0.2, len(coords_df))
            response_labels = (response_scores > np.median(response_scores)).astype(int)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings_np, response_labels, test_size=0.3, random_state=42, stratify=response_labels
            )

            # Model comparison
            models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
            }

            results = []
            for name, model in models.items():
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

                # Test performance
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                test_auc = roc_auc_score(y_test, y_prob)

                results.append({
                    'Model': name,
                    'CV_AUC_mean': cv_scores.mean(),
                    'CV_AUC_std': cv_scores.std(),
                    'Test_AUC': test_auc,
                    'Test_Accuracy': (y_pred == y_test).mean()
                })

            results_df = pd.DataFrame(results)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Model Performance Comparison:**")
                st.dataframe(results_df.round(3), use_container_width=True)

                # Feature importance (for Random Forest)
                rf_model = models['Random Forest']
                rf_model.fit(X_train, y_train)
                feature_importance = rf_model.feature_importances_

                importance_df = pd.DataFrame({
                    'Embedding_Dimension': range(len(feature_importance)),
                    'Importance': feature_importance
                })

                fig = px.bar(
                    importance_df, x='Embedding_Dimension', y='Importance',
                    title="Feature Importance (Random Forest)"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # ROC curves
                from sklearn.metrics import roc_curve

                fig = go.Figure()

                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    auc = roc_auc_score(y_test, y_prob)

                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{name} (AUC = {auc:.3f})'
                    ))

                # Add diagonal line
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Random Classifier'
                ))

                fig.update_layout(
                    title="ROC Curves",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

    with app_tabs[3]:
        if track == "🧬 Biology-Focused Track":
            st.markdown("### Clinical Translation Pathway")

            st.markdown("""
            <div class="biological-insight">
            <h4>🏥 From Research to Clinical Practice</h4>

            <strong>Translation Timeline (5-10 years):</strong>

            **Phase 1: Method Development (Years 1-2)**
            • Algorithm optimization on research datasets
            • Validation on multiple cancer types
            • Collaboration with clinical partners

            **Phase 2: Clinical Validation (Years 3-4)**
            • Retrospective studies on patient samples
            • Prospective pilot studies
            • Regulatory guidance meetings

            **Phase 3: Clinical Trials (Years 5-7)**
            • Multi-center validation studies
            • FDA/EMA submission process
            • Health economics evaluation

            **Phase 4: Implementation (Years 8-10)**
            • Clinical workflow integration
            • Training and education programs
            • Real-world evidence collection
            </div>
            """, unsafe_allow_html=True)

            # Implementation roadmap
            roadmap_data = {
                'Phase': ['Research', 'Validation', 'Clinical Trials', 'Implementation'],
                'Duration_Years': [2, 2, 3, 3],
                'Key_Milestones': [
                    'Algorithm optimization, Publication',
                    'Clinical validation, Regulatory guidance',
                    'Multi-center trials, FDA approval',
                    'Clinical integration, Training'
                ],
                'Success_Probability': [0.9, 0.7, 0.5, 0.8],
                'Cost_Millions': [2, 5, 20, 10]
            }

            roadmap_df = pd.DataFrame(roadmap_data)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    roadmap_df, x='Phase', y='Duration_Years',
                    title="Clinical Translation Timeline",
                    color='Success_Probability',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    roadmap_df, x='Duration_Years', y='Cost_Millions',
                    size='Success_Probability', hover_name='Phase',
                    title="Cost vs Duration by Phase"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Regulatory considerations
            st.markdown("""
            <div class="biological-insight">
            <h4>📋 Regulatory and Implementation Challenges</h4>

            **FDA/EMA Requirements:**
            • Analytical validation (accuracy, precision, reproducibility)
            • Clinical validation (clinical utility, patient outcomes)
            • Quality management system (ISO 13485)
            • Post-market surveillance plan

            **Clinical Workflow Integration:**
            • Electronic health record integration
            • Laboratory information system compatibility  
            • Staff training and certification
            • Quality control procedures

            **Health Economics:**
            • Cost-effectiveness analysis
            • Budget impact modeling
            • Reimbursement strategy
            • Value proposition development
            </div>
            """, unsafe_allow_html=True)

        else:  # Math track
            st.markdown("### Performance Validation Framework")

            st.markdown("""
            <div class="math-insight">
            <h4>📈 Comprehensive Validation Metrics</h4>

            <strong>Clinical Performance Metrics:</strong>

            **Diagnostic Accuracy:**
            • Sensitivity: TP/(TP+FN) - ability to detect positive cases
            • Specificity: TN/(TN+FP) - ability to identify negative cases  
            • PPV: TP/(TP+FP) - probability positive prediction is correct
            • NPV: TN/(TN+FN) - probability negative prediction is correct

            **Discrimination:**
            • AUC-ROC: Area under receiver operating characteristic curve
            • C-index: Concordance index for survival data
            • Net reclassification improvement (NRI)

            **Calibration:**
            • Hosmer-Lemeshow test: goodness of fit
            • Calibration slope: relationship between predicted and observed
            • Calibration intercept: systematic bias assessment
            </div>
            """, unsafe_allow_html=True)

            # Comprehensive validation analysis
            np.random.seed(42)

            # Simulate validation study results
            validation_metrics = {
                'Metric': [
                    'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Accuracy',
                    'AUC-ROC', 'Precision', 'Recall', 'F1-Score'
                ],
                'Training_Set': [0.85, 0.82, 0.78, 0.88, 0.83, 0.88, 0.78, 0.85, 0.81],
                'Validation_Set': [0.82, 0.79, 0.75, 0.85, 0.80, 0.85, 0.75, 0.82, 0.78],
                'Test_Set': [0.80, 0.77, 0.72, 0.83, 0.78, 0.82, 0.72, 0.80, 0.76],
                'CI_Lower': [0.75, 0.72, 0.67, 0.78, 0.73, 0.77, 0.67, 0.75, 0.71],
                'CI_Upper': [0.85, 0.82, 0.77, 0.88, 0.83, 0.87, 0.77, 0.85, 0.81]
            }

            metrics_df = pd.DataFrame(validation_metrics)

            # Performance visualization
            fig = go.Figure()

            metrics_subset = ['Sensitivity', 'Specificity', 'PPV', 'NPV', 'AUC-ROC']
            subset_df = metrics_df[metrics_df['Metric'].isin(metrics_subset)]

            # Training set
            fig.add_trace(go.Scatter(
                x=subset_df['Metric'],
                y=subset_df['Training_Set'],
                mode='markers+lines',
                name='Training Set',
                line=dict(color='blue')
            ))

            # Validation set
            fig.add_trace(go.Scatter(
                x=subset_df['Metric'],
                y=subset_df['Validation_Set'],
                mode='markers+lines',
                name='Validation Set',
                line=dict(color='orange')
            ))

            # Test set with confidence intervals
            fig.add_trace(go.Scatter(
                x=subset_df['Metric'],
                y=subset_df['Test_Set'],
                mode='markers+lines',
                name='Test Set',
                line=dict(color='red'),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=subset_df['CI_Upper'] - subset_df['Test_Set'],
                    arrayminus=subset_df['Test_Set'] - subset_df['CI_Lower']
                )
            ))

            fig.update_layout(
                title="Performance Across Validation Sets",
                xaxis_title="Metric",
                yaxis_title="Performance",
                yaxis=dict(range=[0.6, 1.0]),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statistical significance testing
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Performance Degradation Analysis:**")

                degradation_data = []
                for metric in metrics_subset:
                    row = subset_df[subset_df['Metric'] == metric].iloc[0]
                    train_perf = row['Training_Set']
                    test_perf = row['Test_Set']
                    degradation = (train_perf - test_perf) / train_perf * 100

                    degradation_data.append({
                        'Metric': metric,
                        'Degradation_Percent': degradation,
                        'Acceptable': 'Yes' if degradation < 10 else 'No'
                    })

                degradation_df = pd.DataFrame(degradation_data)
                st.dataframe(degradation_df.round(1), use_container_width=True)

            with col2:
                st.markdown("**Power Analysis:**")

                # Sample size calculation for validation
                from scipy import stats

                alpha = 0.05
                power = 0.80
                effect_size = 0.15  # Minimum clinically important difference

                # Two-sample comparison
                n_per_group = stats.ttest_ind_solve_power(
                    effect_size=effect_size,
                    power=power,
                    alpha=alpha
                )

                st.metric("Required Sample Size", f"{int(np.ceil(n_per_group)):,} per group")
                st.metric("Total Study Size", f"{int(np.ceil(2 * n_per_group)):,} patients")
                st.metric("Power", f"{power:.0%}")
                st.metric("Significance Level", f"{alpha:.1%}")

    # Summary and next steps
    st.markdown('<div class="section-header">🚀 Key Takeaways and Next Steps</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if track == "🧬 Biology-Focused Track":
            st.markdown("""
            <div class="biological-insight">
            <h4>🔬 For Biologists and Clinicians</h4>

            <strong>Key Insights:</strong>
            • Spatial context reveals biology invisible to bulk methods
            • Graph neural networks capture tissue architecture patterns
            • Multiple applications from drug discovery to treatment prediction
            • Clinical translation requires rigorous validation

            <strong>Getting Started:</strong>
            1. Start with existing spatial transcriptomics data
            2. Use simple GCN implementations first
            3. Focus on biological interpretation of results
            4. Collaborate with computational experts
            5. Plan validation studies early

            <strong>Resources:</strong>
            • Scanpy for spatial transcriptomics analysis
            • PyTorch Geometric for graph neural networks
            • Spatial transcriptomics consortiums and databases
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="math-insight">
            <h4>🔢 For Computational Scientists</h4>

            <strong>Mathematical Foundations Mastered:</strong>
            • Spectral graph theory and graph Laplacians
            • Message passing neural network frameworks
            • Optimization on irregular graph structures
            • Statistical validation for spatial data

            <strong>Implementation Skills:</strong>
            1. Graph construction from spatial coordinates
            2. Multiple GNN architectures (GCN, GAT, GraphSAGE, Transformers)
            3. Performance evaluation and comparison
            4. Interpretability analysis methods
            5. Clinical validation frameworks

            <strong>Advanced Topics to Explore:</strong>
            • Heterogeneous graph neural networks
            • Temporal dynamics modeling
            • Multi-modal data integration
            • Causal inference on graphs
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="cs-concept">
        <h4>🌍 Impact and Future Directions</h4>

        <strong>Current Impact:</strong>
        • >50 papers on spatial transcriptomics + GNNs (2020-2024)
        • Multiple clinical validation studies underway
        • Commercial spatial analysis platforms emerging
        • Regulatory guidance development (FDA/EMA)

        <strong>Future Innovations:</strong>
        • Real-time spatial analysis during surgery
        • 3D tissue reconstruction and analysis
        • Multi-patient federated learning
        • Integration with electronic health records

        <strong>Broader Applications:</strong>
        • Neuroscience (brain tissue organization)
        • Developmental biology (organ formation)
        • Immunology (immune system architecture)
        • Drug development (compound screening)
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")

if track == "🧬 Biology-Focused Track":
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🧬 <strong>Congratulations!</strong> You've completed the biology-focused track</p>
        <p>💡 You now understand how graph neural networks revolutionize spatial biology research</p>
        <p>🚀 Ready to apply these methods to your own research questions?</p>
        <p><em>Consider exploring the mathematics track for deeper algorithmic insights</em></p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔢 <strong>Excellent!</strong> You've mastered the mathematical foundations</p>
        <p>💻 You can now implement and optimize graph neural networks for spatial data</p>
        <p>🔬 Ready to tackle cutting-edge spatial biology challenges?</p>
        <p><em>Consider exploring the biology track for domain-specific applications</em></p>
    </div>
    """, unsafe_allow_html=True)