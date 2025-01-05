import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Data Analysis Report Generator", layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

def generate_visualizations(df, target_column):
    """Generate various plots and return them as bytes objects"""
    plots = {}
    
    # Create a figure for distribution plot
    plt.figure(figsize=(10, 6))
    if df[target_column].dtype in ['int64', 'float64']:
        sns.histplot(data=df, x=target_column)
    else:
        sns.countplot(data=df, x=target_column)
    plt.title(f'Distribution of {target_column}')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    plots['distribution'] = buf
    
    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        plots['correlation'] = buf
    
    return plots

def create_pdf_report(df, target_column, report_info, plots):
    """Generate PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title Page
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph(report_info['report_title'], title_style))
    story.append(Paragraph(f"Prepared By: {report_info['prepared_by']}", styles['Normal']))
    story.append(Paragraph(f"Prepared For: {report_info['prepared_for']}", styles['Normal']))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
    story.append(Paragraph(f"Version: {report_info['version']}", styles['Normal']))
    story.append(Spacer(1, 30))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading1']))
    story.append(Paragraph(report_info['purpose'], styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Data Overview
    story.append(Paragraph("Data Overview", styles['Heading1']))
    data_info = [
        ["Number of Rows", str(len(df))],
        ["Number of Columns", str(len(df.columns))],
        ["Target Column", target_column],
        ["Missing Values", str(df.isnull().sum().sum())]
    ]
    t = Table(data_info)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Visualizations
    story.append(Paragraph("Data Visualizations", styles['Heading1']))
    for plot_name, plot_buf in plots.items():
        plot_buf.seek(0)
        img = Image(plot_buf, width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 20))
    
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
