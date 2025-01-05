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
        'CustomBody': {  # Renamed from BodyText to avoid conflict
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

def create_pdf_report(df, target_column, report_info, plots, analysis_results):
    """Generate detailed PDF report"""
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
    
    # Executive Summary
    story.append(Paragraph("1. Executive Summary", styles['ChapterTitle']))
    story.append(Paragraph(report_info['purpose'], styles['CustomBody']))
    story.append(PageBreak())
    
    # Introduction
    story.append(Paragraph("2. Introduction", styles['ChapterTitle']))
    story.append(Paragraph("2.1 Objective", styles['SectionTitle']))
    story.append(Paragraph(report_info['purpose'], styles['CustomBody']))
    
    # Add dataset overview
    story.append(Paragraph("2.2 Dataset Overview", styles['SectionTitle']))
    overview_text = f"""
    This analysis examines a dataset containing {len(df)} records with {len(df.columns)} features.
    The primary target variable for analysis is '{target_column}'.
    The dataset includes {len(df.select_dtypes(include=['int64', 'float64']).columns)} numerical and
    {len(df.select_dtypes(include=['object']).columns)} categorical variables.
    """
    story.append(Paragraph(overview_text, styles['CustomBody']))
    story.append(PageBreak())
    
    # Data Overview
    story.append(Paragraph("3. Data Overview", styles['ChapterTitle']))
    
    # Create data summary table
    data_info = [
        ["Metric", "Value"],
        ["Number of Records", str(len(df))],
        ["Number of Features", str(len(df.columns))],
        ["Missing Values", str(df.isnull().sum().sum())],
        ["Numeric Features", str(len(df.select_dtypes(include=['int64', 'float64']).columns))],
        ["Categorical Features", str(len(df.select_dtypes(include=['object']).columns))]
    ]
    
    # Add the table with styling
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ])
    
    t = Table(data_info, colWidths=[4*inch, 2*inch])
    t.setStyle(table_style)
    story.append(t)
    
    # Add visualizations
    story.append(Paragraph("4. Visualizations", styles['ChapterTitle']))
    for plot_name, plot_buf in plots.items():
        story.append(Paragraph(f"4.{plot_name.capitalize()}", styles['SectionTitle']))
        plot_buf.seek(0)
        img = Image(plot_buf, width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 20))
    
    # Statistical Analysis
    if isinstance(analysis_results.get('basic_stats'), pd.Series):
        story.append(Paragraph("5. Statistical Analysis", styles['ChapterTitle']))
        story.append(Paragraph("5.1 Basic Statistics", styles['SectionTitle']))
        
        # Convert basic stats to table
        stats_data = [["Metric", "Value"]]
        for stat, value in analysis_results['basic_stats'].items():
            stats_data.append([stat, f"{value:.2f}" if isinstance(value, float) else str(value)])
        
        t = Table(stats_data, colWidths=[4*inch, 2*inch])
        t.setStyle(table_style)
        story.append(t)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# Set page config
st.set_page_config(page_title="Data Analysis Report Generator", layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# App title
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
        
        st.session_state.df = df
        
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
                plots = generate_advanced_visualizations(df, target_column)
                
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
                
                # Display interactive visualizations in Streamlit
                st.header("Interactive Visualizations")
                
                # Distribution plot using Plotly
                if df[target_column].dtype in ['int64', 'float64']:
                    fig = px.histogram(df, x=target_column, 
                                     title=f'Distribution of {target_column}',
                                     marginal="box")  # Added box plot on the margin
                    st.plotly_chart(fig)
                    
                    # Additional statistics
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
                
                # Correlation heatmap using Plotly
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 1:
                    st.header("Correlation Analysis")
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix,
                                  title='Correlation Heatmap',
                                  color_continuous_scale='RdBu')
                    st.plotly_chart(fig)
                    
                    # Highlight strong correlations
                    st.subheader("Strong Correlations")
                    strong_corr = (corr_matrix.abs() > 0.5) & (corr_matrix != 1.000)
                    if strong_corr.any().any():
                        for idx, row in enumerate(strong_corr.index):
                            for col in strong_corr.columns[idx+1:]:
                                if strong_corr.loc[row, col]:
                                    st.write(f"• {row} vs {col}: {corr_matrix.loc[row, col]:.3f}")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Please check your input file and try again.")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ by Your Data Team")
