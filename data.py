import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def load_data(file):
    """Load data from different file formats"""
    try:
        file_extension = file.name.split('.')[-1].lower()

        if file_extension == 'csv':
            df = pd.read_csv(file)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file)
        elif file_extension == 'json':
            df = pd.read_json(file)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or JSON file.")
            return None

        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def clean_data(df):
    """Clean and prepare data"""
    # Convert string numbers to float where possible
    for column in df.columns:
        if df[column].dtype == 'object':
            # Try to convert to numeric, if fails keep as categorical
            try:
                numeric_values = pd.to_numeric(df[column], errors='coerce')
                # Only convert if most values are numeric (>50%)
                if numeric_values.notna().sum() / len(numeric_values) > 0.5:
                    df[column] = numeric_values
            except:
                continue

    return df

def handle_null_values(df, method, custom_value=None):
    """Handle null values in the dataframe"""
    if method == "Mean":
        df.fillna(df.mean(), inplace=True)
    elif method == "Median":
        df.fillna(df.median(), inplace=True)
    elif method == "Mode":
        df.fillna(df.mode().iloc[0], inplace=True)
    elif method == "Custom Value" and custom_value is not None:
        df.fillna(custom_value, inplace=True)
    return df

def get_numeric_columns(df):
    """Get list of numeric columns"""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df):
    """Get list of categorical columns"""
    return df.select_dtypes(exclude=[np.number]).columns.tolist()

def prepare_categorical_plot_data(df, x_column, y_column, aggregation='mean'):
    """Prepare data for categorical plots by aggregating numerical values"""
    if df[x_column].dtype == 'object' and df[y_column].dtype != 'object':
        # Aggregate numerical values for each category
        if aggregation == 'mean':
            agg_df = df.groupby(x_column)[y_column].mean().reset_index()
        elif aggregation == 'sum':
            agg_df = df.groupby(x_column)[y_column].sum().reset_index()
        elif aggregation == 'count':
            agg_df = df.groupby(x_column)[y_column].count().reset_index()
        return agg_df
    return df

def create_visualization(df, chart_type, x_column, y_column, color_column=None, aggregation='mean'):
    """Create different types of visualizations"""
    try:
        # Prepare data based on column types
        if x_column and y_column:  # For plots requiring both x and y
            plot_df = prepare_categorical_plot_data(df, x_column, y_column, aggregation)
        else:  # For plots requiring only one variable (like histograms)
            plot_df = df

        if chart_type == "Scatter Plot":
            fig = px.scatter(plot_df, x=x_column, y=y_column, color=color_column,
                           title=f"{x_column} vs {y_column}")

        elif chart_type == "Line Plot":
            fig = px.line(plot_df, x=x_column, y=y_column, color=color_column,
                         title=f"{x_column} vs {y_column}")

        elif chart_type == "Bar Plot":
            fig = px.bar(plot_df, x=x_column, y=y_column, color=color_column,
                        title=f"{x_column} vs {y_column}")

        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_column, y=y_column, color=color_column,
                        title=f"Box Plot of {y_column} by {x_column}")

        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_column, color=color_column,
                             title=f"Histogram of {x_column}")

        elif chart_type == "Pie Chart":
            # For pie charts, always aggregate the values
            agg_df = df.groupby(x_column)[y_column].sum().reset_index()
            fig = px.pie(agg_df, values=y_column, names=x_column,
                        title=f"Pie Chart of {y_column} by {x_column}")

        elif chart_type == "Count Plot":
            # Count plot for categorical variables
            count_df = df[x_column].value_counts().reset_index()
            count_df.columns = [x_column, 'count']
            fig = px.bar(count_df, x=x_column, y='count',
                        title=f"Count Plot of {x_column}")

        # Update layout
        fig.update_layout(
            template="plotly_white",
            xaxis_title=x_column,
            yaxis_title=y_column if y_column else "Count",
            xaxis={'categoryorder':'total descending'} if df[x_column].dtype == 'object' else None
        )

        # Rotate x-axis labels if categorical
        if df[x_column].dtype == 'object':
            fig.update_xaxes(tickangle=45)

        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def show_data_insights(df):
    """Show basic data insights"""
    st.subheader("Data Insights")

    # Basic information
    st.write("Dataset Shape:", df.shape)

    # Data types
    st.write("Data Types:")
    st.write(df.dtypes)

    # Summary statistics for numeric columns
    numeric_cols = get_numeric_columns(df)
    if numeric_cols:
        st.write("Summary Statistics (Numeric Columns):")
        st.write(df[numeric_cols].describe())

    # Categorical columns analysis
    categorical_cols = get_categorical_columns(df)
    if categorical_cols:
        st.write("Categorical Columns Analysis:")
        for col in categorical_cols:
            st.write(f"\nUnique values in {col}:")
            st.write(df[col].value_counts().head())

    # Missing values
    st.write("Missing Values:")
    st.write(df.isnull().sum())

def main():
    st.set_page_config(page_title="Data Analysis Tool", layout="wide")

    st.title("Interactive Data Analysis Tool")
    st.write("Upload your data file (CSV, Excel, or JSON) to analyze and visualize patterns")

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'json'])

    if uploaded_file is not None:
        # Load and clean data
        df = load_data(uploaded_file)

        if df is not None:
            df = clean_data(df)

            # Show raw data
            st.subheader("Raw Data Preview")
            st.write(df.head())

            # Data insights
            show_data_insights(df)

            # Data cleaning options
            st.subheader("Data Cleaning")
            st.write("Number of Null Values in Each Column:")
            st.write(df.isnull().sum())

            null_method = st.selectbox("Handle Null Values", ["None", "Mean", "Median", "Mode", "Custom Value"])
            if null_method == "Custom Value":
                custom_value = st.text_input("Enter custom value")
            else:
                custom_value = None
            if st.button("Apply Null Handling"):
                df = handle_null_values(df, null_method, custom_value)
                st.write("Null values handled successfully!")

            # Visualization options
            st.subheader("Data Visualization")

            col1, col2, col3 = st.columns(3)

            with col1:
                chart_type = st.selectbox(
                    "Select Chart Type",
                    ["Bar Plot", "Scatter Plot", "Line Plot", "Box Plot",
                     "Histogram", "Pie Chart", "Count Plot"]
                )

            with col2:
                numeric_columns = get_numeric_columns(df)
                categorical_columns = get_categorical_columns(df)
                all_columns = df.columns.tolist()

                if chart_type == "Count Plot":
                    x_column = st.selectbox("Select Category", categorical_columns)
                    y_column = None
                elif chart_type == "Histogram":
                    x_column = st.selectbox("Select Column", all_columns)
                    y_column = None
                else:
                    x_column = st.selectbox("Select X-axis", all_columns)
                    if chart_type == "Pie Chart":
                        y_column = st.selectbox("Select Values", numeric_columns)
                    else:
                        y_column = st.selectbox("Select Y-axis", numeric_columns)

            with col3:
                if chart_type not in ["Pie Chart", "Count Plot"]:
                    color_column = st.selectbox("Select Color Column (optional)",
                                              ["None"] + categorical_columns)
                    color_column = None if color_column == "None" else color_column

                    if x_column in categorical_columns and y_column in numeric_columns:
                        aggregation = st.selectbox(
                            "Select Aggregation Method",
                            ["mean", "sum", "count"],
                            help="How to aggregate numerical values for each category"
                        )
                    else:
                        aggregation = "mean"
                else:
                    color_column = None
                    aggregation = "sum"

            # Create and display visualization
            fig = create_visualization(df, chart_type, x_column, y_column,
                                    color_column, aggregation)

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            # Data filtering
            st.subheader("Data Filtering")

            # Column selection
            selected_columns = st.multiselect(
                "Select columns to display",
                df.columns.tolist(),
                default=df.columns.tolist()
            )

            # Text search
            search_term = st.text_input("Search in data")

            # Filter data based on selection and search
            filtered_df = df[selected_columns]
            if search_term:
                filtered_df = filtered_df[
                    filtered_df.astype(str).apply(
                        lambda x: x.str.contains(search_term, case=False)
                    ).any(axis=1)
                ]

            # Show filtered data
            st.write("Filtered Data:")
            st.write(filtered_df)

            # Download filtered data
            st.download_button(
                label="Download filtered data as CSV",
                data=filtered_df.to_csv(index=False).encode('utf-8'),
                file_name='filtered_data.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
