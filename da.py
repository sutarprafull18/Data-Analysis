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
    
    # Title style
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    ))
    
    # Chapter style
    styles.add(ParagraphStyle(
        name='ChapterTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=25,
        spaceBefore=30
    ))
    
    # Section style
    styles.add(ParagraphStyle(
        name='SectionTitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=20,
        spaceBefore=20
    ))
    
    # Body text style
    styles.add(ParagraphStyle(
        name='BodyText',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        alignment=TA_JUSTIFY
    ))
    
    return styles

def generate_advanced_visualizations(df, target_column):
    """Generate comprehensive visualizations for the report"""
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
        
        # Pairplot for numerical variables
        if len(numeric_cols) <= 5:  # Limit to prevent overcrowding
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
    """Perform comprehensive statistical analysis"""
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
    """Generate detailed PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=72)
    
    styles = create_custom_styles()
    story = []
    
    # Title Page
    story.append(Paragraph(report_info['report_title'], styles['CustomTitle']))
    story.append(Spacer(1, 30))
    story.append(Paragraph(f"Prepared By: {report_info['prepared_by']}", styles['BodyText']))
    story.append(Paragraph(f"Prepared For: {report_info['prepared_for']}", styles['BodyText']))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['BodyText']))
    story.append(Paragraph(f"Version: {report_info['version']}", styles['BodyText']))
    story.append(PageBreak())
    
    # Table of Contents (placeholder - would need more complex logic for actual TOC)
    story.append(Paragraph("Table of Contents", styles['ChapterTitle']))
    # Add TOC items here
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['ChapterTitle']))
    story.append(Paragraph(report_info['purpose'], styles['BodyText']))
    story.append(Paragraph("Key Findings:", styles['SectionTitle']))
    # Add key findings based on analysis
    story.append(PageBreak())
    
    # Introduction
    story.append(Paragraph("1. Introduction", styles['ChapterTitle']))
    story.append(Paragraph("1.1 Objective", styles['SectionTitle']))
    story.append(Paragraph(report_info['purpose'], styles['BodyText']))
    story.append(Paragraph("1.2 Scope", styles['SectionTitle']))
    story.append(Paragraph("This analysis covers the following aspects:", styles['BodyText']))
    story.append(PageBreak())
    
    # Methodology
    story.append(Paragraph("2. Methodology", styles['ChapterTitle']))
    story.append(Paragraph("2.1 Data Collection", styles['SectionTitle']))
    story.append(Paragraph("2.2 Data Preprocessing", styles['SectionTitle']))
    story.append(Paragraph("2.3 Analysis Techniques", styles['SectionTitle']))
    story.append(PageBreak())
    
    # Data Overview
    story.append(Paragraph("3. Data Overview", styles['ChapterTitle']))
    data_info = [
        ["Dataset Properties", "Value"],
        ["Number of Records", str(len(df))],
        ["Number of Features", str(len(df.columns))],
        ["Target Variable", target_column],
        ["Missing Values", str(df.isnull().sum().sum())],
        ["Time Period", "N/A"],  # Would need to be determined from data
    ]
    t = Table(data_info, colWidths=[4*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t)
    story.append(PageBreak())
    
    # Statistical Analysis
    story.append(Paragraph("4. Statistical Analysis", styles['ChapterTitle']))
    story.append(Paragraph("4.1 Descriptive Statistics", styles['SectionTitle']))
    # Add descriptive statistics table
    story.append(Paragraph("4.2 Distribution Analysis", styles['SectionTitle']))
    for plot_name, plot_buf in plots.items():
        plot_buf.seek(0)
        img = Image(plot_buf, width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 20))
    story.append(PageBreak())
    
    # Findings and Insights
    story.append(Paragraph("5. Findings and Insights", styles['ChapterTitle']))
    story.append(Paragraph("5.1 Key Patterns", styles['SectionTitle']))
    story.append(Paragraph("5.2 Anomalies", styles['SectionTitle']))
    story.append(Paragraph("5.3 Trends", styles['SectionTitle']))
    story.append(PageBreak())
    
    # Recommendations
    story.append(Paragraph("6. Recommendations", styles['ChapterTitle']))
    # Add recommendations based on analysis
    story.append(PageBreak())
    
    # Limitations and Assumptions
    story.append(Paragraph("7. Limitations and Assumptions", styles['ChapterTitle']))
    story.append(PageBreak())
    
    # Appendix
    story.append(Paragraph("Appendix", styles['ChapterTitle']))
    story.append(Paragraph("A.1 Data Dictionary", styles['SectionTitle']))
    story.append(Paragraph("A.2 Detailed Statistical Results", styles['SectionTitle']))
    story.append(Paragraph("A.3 Methodology Details", styles['SectionTitle']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# App title
st.title("📊 Data Analysis Report Generator")

# Sidebar for inputs
with st.sidebar:
    st.header("Report Information")
    report_info = {
        'report_title': st.text_input("Report Title", "Data Analysis Report"),
        'prepared_by': st.text_input("Prepared By", "Data Analyst"),
        'prepared_for': st.text_input("Prepared For", "Organization"),
        'version': st.text_input("Version", "1.0"),
        'purpose': st.text_area("Purpose of Analysis", "This analysis aims to...")
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
        
        # Select target column
        target_column = st.selectbox("Select Target Feature", df.columns.tolist())
        
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                # Generate visualizations
                plots = generate_visualizations(df, target_column)
                
                # Create PDF
                pdf_buffer = create_pdf_report(df, target_column, report_info, plots)
                
                # Offer download
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
                
                # Display interactive visualizations in Streamlit
                st.header("Interactive Visualizations")
                
                # Distribution plot using Plotly
                if df[target_column].dtype in ['int64', 'float64']:
                    fig = px.histogram(df, x=target_column, title=f'Distribution of {target_column}')
                else:
                    value_counts = df[target_column].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=f'Distribution of {target_column}')
                st.plotly_chart(fig)
                
                # Correlation heatmap using Plotly
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, 
                                  title='Correlation Heatmap',
                                  color_continuous_scale='RdBu')
                    st.plotly_chart(fig)
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
