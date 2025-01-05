import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, 
                              TableStyle, Image, PageBreak, ListFlowable, ListItem)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
import io
from datetime import datetime
import plotly.express as px
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Set page config
st.set_page_config(page_title="Data Analysis Report Generator", layout="wide")

def create_custom_styles():
    """Create custom styles for the PDF report"""
    styles = getSampleStyleSheet()
    
    # Dictionary of style definitions
    custom_styles = {
        'CustomTitle': {
            'parent': styles['Heading1'],
            'fontSize': 24,
            'spaceAfter': 30,
            'alignment': TA_CENTER
        },
        'ChapterTitle': {
            'parent': styles['Heading1'],
            'fontSize': 20,
            'spaceAfter': 25,
            'spaceBefore': 30
        },
        'SectionTitle': {
            'parent': styles['Heading2'],
            'fontSize': 16,
            'spaceAfter': 20,
            'spaceBefore': 20
        },
        'CustomBody': {
            'parent': styles['Normal'],
            'fontSize': 12,
            'leading': 16,
            'alignment': TA_JUSTIFY
        }
    }
    
    # Add styles if they don't exist
    for style_name, style_props in custom_styles.items():
        if style_name not in styles:
            styles.add(ParagraphStyle(
                name=style_name,
                parent=style_props['parent'],
                **{k: v for k, v in style_props.items() if k != 'parent'}
            ))
    
    return styles

def generate_visualizations(df, target_column):
    """Generate visualizations for the report"""
    plots = {}
    
    # Distribution plot
    plt.figure(figsize=(10, 6))
    if df[target_column].dtype in ['int64', 'float64']:
        sns.histplot(data=df, x=target_column, kde=True)
        plt.title(f'Distribution of {target_column}')
    else:
        sns.countplot(data=df, x=target_column)
        plt.xticks(rotation=45)
        plt.title(f'Frequency Distribution of {target_column}')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    plots['distribution'] = buf
    
    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        plots['correlation'] = buf
        
        # Pairplot for numerical variables (limited to prevent overcrowding)
        if len(numeric_cols) <= 5:
            plt.figure(figsize=(15, 15))
            sns.pairplot(df[numeric_cols])
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            plt.close()
            plots['pairplot'] = buf
    
    # Box plots for outlier detection
    if df[target_column].dtype in ['int64', 'float64']:
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=df[target_column])
        plt.title(f'Box Plot of {target_column}')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        plots['boxplot'] = buf
    
    return plots

def perform_statistical_analysis(df, target_column):
    """Perform statistical analysis on the data"""
    analysis_results = {}
    
    # Basic statistics
    analysis_results['basic_stats'] = df[target_column].describe()
    
    if df[target_column].dtype in ['int64', 'float64']:
        # Normality test
        stat, p_value = stats.normaltest(df[target_column].dropna())
        analysis_results['normality_test'] = {
            'statistic': stat,
            'p_value': p_value
        }
        
        # Skewness and Kurtosis
        analysis_results['skewness'] = stats.skew(df[target_column].dropna())
        analysis_results['kurtosis'] = stats.kurtosis(df[target_column].dropna())
    
    # Missing values analysis
    analysis_results['missing_values'] = df[target_column].isnull().sum()
    analysis_results['missing_percentage'] = (df[target_column].isnull().sum() / len(df)) * 100
    
    return analysis_results

def create_pdf_report(df, target_column, report_info, plots, analysis_results):
    """Generate PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=72)
    
    styles = create_custom_styles()
    story = []
    
    # Title Page
    story.append(Paragraph(report_info['report_title'], styles['CustomTitle']))
    story.append(Spacer(1, 30))
    story.append(Paragraph(f"Prepared By: {report_info['prepared_by']}", styles['CustomBody']))
    story.append(Paragraph(f"Prepared For: {report_info['prepared_for']}", styles['CustomBody']))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['CustomBody']))
    story.append(Paragraph(f"Version: {report_info['version']}", styles['CustomBody']))
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", styles['ChapterTitle']))
    toc_items = [
        "1. Executive Summary",
        "2. Introduction",
        "3. Methodology",
        "4. Data Overview",
        "5. Statistical Analysis",
        "6. Findings and Insights",
        "7. Recommendations",
        "8. Limitations and Assumptions",
        "9. Appendix"
    ]
    for item in toc_items:
        story.append(Paragraph(item, styles['CustomBody']))
    story.append(PageBreak())
    
    # Add rest of the sections (as defined in previous create_pdf_report function)
    # [Previous sections code remains the same]
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# Streamlit UI
def main():
    st.title("📊 Comprehensive Data Analysis Report Generator")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Report Information")
        report_info = {
            'report_title': st.text_input("Report Title", "Comprehensive Data Analysis Report"),
            'prepared_by': st.text_input("Prepared By", "Data Analyst"),
            'prepared_for': st.text_input("Prepared For", "Organization"),
            'version': st.text_input("Version", "1.0"),
            'purpose': st.text_area("Purpose of Analysis", 
                "This analysis aims to provide comprehensive insights into the dataset, "
                "identifying key patterns, trends, and actionable recommendations.")
        }

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display data overview
            st.header("Data Preview")
            st.dataframe(df.head())
            
            # Column info
            st.header("Dataset Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Rows", df.shape[0])
            with col2:
                st.metric("Number of Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Select target column
            target_column = st.selectbox("Select Target Feature", df.columns.tolist())
            
            if st.button("Generate Report"):
                with st.spinner("Generating comprehensive report..."):
                    # Generate visualizations
                    plots = generate_visualizations(df, target_column)
                    
                    # Perform statistical analysis
                    analysis_results = perform_statistical_analysis(df, target_column)
                    
                    # Create PDF
                    pdf_buffer = create_pdf_report(df, target_column, report_info, plots, analysis_results)
                    
                    # Offer download
                    st.download_button(
                        label="📥 Download Comprehensive PDF Report",
                        data=pdf_buffer,
                        file_name=f"comprehensive_analysis_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
                    
                    # Display interactive visualizations
                    st.header("Interactive Visualizations")
                    
                    if df[target_column].dtype in ['int64', 'float64']:
                        fig = px.histogram(df, x=target_column, 
                                         title=f'Distribution of {target_column}',
                                         marginal="box")
                        st.plotly_chart(fig)
                        
                        # Statistics summary
                        st.header("Statistical Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean", f"{df[target_column].mean():.2f}")
                        with col2:
                            st.metric("Median", f"{df[target_column].median():.2f}")
                        with col3:
                            st.metric("Std Dev", f"{df[target_column].std():.2f}")
                        with col4:
                            st.metric("Skewness", f"{stats.skew(df[target_column].dropna()):.2f}")
                    
                    # Correlation analysis
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    if len(numeric_cols) > 1:
                        st.header("Correlation Analysis")
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(corr_matrix,
                                      title='Correlation Heatmap',
                                      color_continuous_scale='RdBu')
                        st.plotly_chart(fig)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error("Please check your input file and try again.")

    # Add footer
    st.markdown("---")
    st.markdown("Made with ❤️ by Your Data Team")

if __name__ == "__main__":
    main()
