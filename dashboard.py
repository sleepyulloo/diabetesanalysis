import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.loading import load, replace_zeros_with_median
from src.visualizations import plot_correlation_heatmap, plot_outcome_distribution
from src.analysis import summary, number_of_zeros, missing_data

# Setting up
st.set_page_config(
    page_title="Diabetes Dataset Explorer",
    page_icon="🧪",
    layout="wide"
)

# CSS = better style
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #3366ff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff6633;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# title
st.markdown("<h1 class='main-header'>🧪 Diabetes Dataset Explorer</h1>", unsafe_allow_html=True)


df = load()
columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_clean = replace_zeros_with_median(df, columns)

# organization = tabs 
tab1, tab2 = st.tabs(["📊 Data Overview", "📈 Visualizations"])

with tab1:
    st.markdown("<h2 class='section-header'>Dataset Sample</h2>", unsafe_allow_html=True)
    st.dataframe(df_clean.head(), use_container_width=True)
    
    with st.expander("Show Summary Statistics"):
        st.write(summary(df_clean))
    
    with st.expander("Show Missing Values"):
        st.write(missing_data(df_clean))
    
    with st.expander("Show Zero Value Counts"):
        st.write(number_of_zeros(df_clean))

with tab2:
    st.markdown("<h2 class='section-header'>Feature Distribution Visualizations</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_viz = st.radio(
            "Select Visualization",
            ["Class Distribution", "Feature Boxplots", "Correlation Heatmap", "Custom Heatmap"]
        )
    
    with col2:
        if selected_viz == "Class Distribution":
            st.subheader("Outcome Distribution")
            st.bar_chart(plot_outcome_distribution(df_clean))
            
        elif selected_viz == "Feature Boxplots":
            st.subheader("Feature Distributions (Boxplots)")
            
            features = st.multiselect(
                "Select features to display:",
                options=df_clean.columns.tolist(),
                default=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness']
            )
            
            if features:
                colors = ['pink', 'red', 'lightgreen', 'brown', 'lightblue', 'purple', '#FF7F00', 'teal']
                
                fig, axes = plt.subplots(len(features), 1, figsize=(10, 2*len(features)))
                
                if len(features) == 1:
                    axes = [axes]
                
                for i, feature in enumerate(features):
                    color_idx = i % len(colors)
                    sns.boxplot(x=df_clean[feature], ax=axes[i], color=colors[color_idx])
                    axes[i].set_title(f"{feature} Distribution")
                    
                    if feature == 'Pregnancies':
                        axes[i].set_xticks(range(0, 19, 2))
                    elif feature == 'Glucose':
                        axes[i].set_xticks(range(0, 201, 25))
                    elif feature == 'BloodPressure':
                        axes[i].set_xticks(range(0, 131, 10))
                    elif feature == 'SkinThickness':
                        axes[i].set_xticks(range(0, 101, 10))
                    elif feature == 'Insulin':
                        axes[i].set_xticks(range(0, 901, 100))
                    elif feature == 'BMI':
                        axes[i].set_xticks(range(0, 80, 5))
                    elif feature == 'Age':
                        axes[i].set_xticks(range(0, 101, 10))
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Please select at least one feature.")
            
        elif selected_viz == "Correlation Heatmap":
            st.subheader("Correlation Matrix Heatmap")
            fig = plot_correlation_heatmap(df_clean)
            st.pyplot(fig)
            
        elif selected_viz == "Custom Heatmap":
            st.subheader("Custom Correlation Heatmap")
            
            corr_features = st.multiselect(
                "Select features for correlation analysis:",
                options=df_clean.columns.tolist(),
                default=['Glucose', 'Outcome', 'BMI', 'Insulin', 'DiabetesPedigreeFunction']
            )
            
            if corr_features and len(corr_features) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation = df_clean[corr_features].corr()
                
                cmap_options = {'Viridis': 'viridis', 'Magma': 'magma', 'Vanimo': 'plasma', 'Blues': 'Blues', 'Reds': 'Reds'}
                selected_cmap = st.selectbox("Choose color palette:", options=list(cmap_options.keys()))
                
                sns.heatmap(
                    correlation,
                    annot=True,
                    cmap=cmap_options[selected_cmap],
                    cbar=False,
                    linewidths=2,
                    linecolor='black',
                    ax=ax
                )
                plt.title("Feature Correlation Matrix", fontsize=16)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Please select at least two features for correlation analysis.")

