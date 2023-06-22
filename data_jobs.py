# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# Load data
@st.cache_resource
def load_data():
    data = pd.read_csv('data_science_job.csv')
    data['Salary'] = data['Salary'].str.extract('(\d+)', expand=False)

    # Remove rows with NaN after extraction
    data = data.dropna(subset=['Salary'])
    
    # Convert to float and scale up by 1000
    data['Salary'] = data['Salary'].astype(float) * 1000
    data['Experience level'] = data['Experience level'].apply(map_experience)
    return data

def map_experience(experience_level):
    if experience_level == 'Entry-level':
        return 1
    elif experience_level == 'Mid-level':
        return np.random.randint(1, 5)
    elif experience_level == 'Senior-level':
        return np.random.randint(5, 10)
    elif experience_level == 'Executive-Level':
        return np.random.randint(10, 15)
    else:
        return 0
# Introduction Page
def intro():
    st.title("Welcome to the Data Science Jobs Finder!")
    st.markdown("""
    This application is a tool that can help you find the most suitable data science jobs for you.
    You can navigate through the application to explore job opportunities and discover jobs that match your credentials.
    The application is broken down into the following sections:

    - **Introduction:** This is the page you're currently on. It provides an overview of what the application does and how to use it.

    - **Filter Jobs:** This page allows you to filter the available data science jobs based on company, job title, location, job type, 
      experience level, and salary range. Use the selectors to choose which aspects to filter on, and the application will show you 
      a table of jobs that meet your criteria.

    - **Match Credentials:** On this page, you can input your qualifications, such as your degree, certification, and years of experience. 
      The application will match these with the job requirements to find the jobs that are the best fit for you.

    - **General Statistics:** This page provides useful statistics about the available jobs, such as the distribution of jobs by company, 
      the average salary for different job titles, and more.

    To navigate between these pages, use the selector in the sidebar on the left of the application. Let's find your dream job together!
    """)



# Filter Jobs Page
def filter_jobs(data):
    st.title("Find your dream job here!")

    with st.form(key='filter_form'):
        selected_company = st.multiselect("Company", data['Company'].unique())
        selected_job_title = st.multiselect("Job Title", data['Job Title'].unique())
        selected_location = st.multiselect("Location", data['Location'].unique())
        selected_job_type = st.multiselect("Job Type", data['Job Type'].unique())
        selected_experience_level = st.multiselect("Experience level", data['Experience level'].unique())
        min_salary, max_salary = st.slider("Salary Range($)", int(data['Salary'].min()), int(data['Salary'].max()), (int(data['Salary'].min()), int(data['Salary'].max())))

        filter_button = st.form_submit_button(label='Filter')

    if filter_button:
        filtered_data = data

        if selected_company:
            filtered_data = filtered_data[filtered_data['Company'].isin(selected_company)]
        if selected_job_title:
            filtered_data = filtered_data[filtered_data['Job Title'].isin(selected_job_title)]
        if selected_location:
            filtered_data = filtered_data[filtered_data['Location'].isin(selected_location)]
        if selected_job_type:
            filtered_data = filtered_data[filtered_data['Job Type'].isin(selected_job_type)]
        if selected_experience_level:
            filtered_data = filtered_data[filtered_data['Experience level'].isin(selected_experience_level)]
        if min_salary or max_salary:
            filtered_data = filtered_data[(filtered_data['Salary'] >= min_salary) & (filtered_data['Salary'] <= max_salary)]
        
        st.table(filtered_data)

# Match Credentials Page
def match_credentials(data):
    st.title("Match your credentials here!")
    # Your code for matching credentials page
   
    with st.form(key='match_form'):
        degree_input = st.text_input('Enter your degree')
        cert_input = st.text_input('Enter your skills')
        years_experience_input = st.number_input('Enter your years of experience', min_value=0, value=0)

        match_button = st.form_submit_button(label='Match')

    if match_button:
        data['Experience level'] = data['Experience level'].apply(map_experience)
        
        # Combine 'degree_input' and 'cert_input'
        qualifications = ' '.join([degree_input, cert_input])

        # Vectorize the job requirements and the candidate's qualifications
        tfidf_vectorizer = TfidfVectorizer()
        job_requirements = tfidf_vectorizer.fit_transform(data['Requirment of the company '])

        # Prepare the data for the model
        X = np.column_stack([job_requirements.toarray(), data['Experience level'].values])

        # Train a NearestNeighbors model
        model = NearestNeighbors(n_neighbors=5, metric='cosine')
        model.fit(X)

        # Get the recommendations
        query = tfidf_vectorizer.transform([qualifications])
        query = np.column_stack([query.toarray(), years_experience_input])
        distances, indices = model.kneighbors(query)

        # Display the recommended jobs
        recommended_jobs = data.iloc[indices[0]]
        st.table(recommended_jobs)

# General Statistics Page
# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# General Statistics Page
def stats(data):
    st.title("Get the general stats!")

    # Salary expectations based on roles
    st.subheader("Salary expectations based on roles")
    roles_salary = data.groupby('Job Title')['Salary'].mean().sort_values(ascending=False)
    fig = px.bar(roles_salary, x=roles_salary.index, y='Salary', labels={'x':'Job Title', 'y':'Salary'})
    st.plotly_chart(fig)

    # Salary expectations based on skills
    st.subheader("Salary expectations based on skills")
    skills_salary = data.groupby('Requirment of the company ')['Salary'].mean().sort_values(ascending=False)
    fig = px.bar(skills_salary, x=skills_salary.index, y='Salary', labels={'x':'Skills', 'y':'Salary'})
    st.plotly_chart(fig)

    # Other charts can be added in a similar fashion

    # Interactive selectors for different charts
    chart_select = st.selectbox(
        'Select the Chart',
        ('Top 10 Highest Paying Companies', 'Top 10 Highest Paying Job Titles', 'Salary Distribution'))

    if chart_select == 'Top 10 Highest Paying Companies':
        top_companies = data.groupby('Company')['Salary'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(top_companies, x=top_companies.index, y='Salary', labels={'x':'Company', 'y':'Salary'})
        st.plotly_chart(fig)
        
    elif chart_select == 'Top 10 Highest Paying Job Titles':
        top_jobs = data.groupby('Job Title')['Salary'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(top_jobs, x=top_jobs.index, y='Salary', labels={'x':'Job Title', 'y':'Salary'})
        st.plotly_chart(fig)
        
    elif chart_select == 'Salary Distribution':
        fig = px.histogram(data, x="Salary")
        st.plotly_chart(fig)


def main():
    data = load_data()

    pages = {
        "Introduction": intro,
        "Filter Jobs": lambda: filter_jobs(data),
        "Match Credentials": lambda: match_credentials(data),
        "General Statistics": lambda: stats(data)
    }

    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Select a page:', tuple(pages.keys()))

    # Call the function corresponding to the selected page
    pages[page]()

if __name__ == "__main__":
    main()
