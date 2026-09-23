import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page layout configuration
st.set_page_config(page_title="AI Travel Recommender", page_icon="✈️", layout="wide")

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('traveler-trip-data.csv')

    # Clean missing values
    df['Destination'] = df['Destination'].fillna('Unknown')
    df['Traveler age'] = df['Traveler age'].fillna(df['Traveler age'].median())
    df['Accommodation cost'] = pd.to_numeric(df['Accommodation cost'], errors='coerce').fillna(0)
    df['Transportation cost'] = pd.to_numeric(df['Transportation cost'], errors='coerce').fillna(0)
    df['Duration (days)'] = df['Duration (days)'].fillna(df['Duration (days)'].median())
    df['Accommodation type'] = df['Accommodation type'].fillna('Unknown')
    df['Transportation type'] = df['Transportation type'].fillna('Unknown')

    # Destination type mapping with safe lookup fallback
    destination_type_map = {
        'London, UK': 'city', 'Phuket, Thailand': 'beach', 'Bali, Indonesia': 'beach',
        'New York, USA': 'city', 'Tokyo, Japan': 'city', 'Paris, France': 'city',
        'Sydney, Australia': 'beach', 'Rio de Janeiro, Brazil': 'beach',
        'Amsterdam, Netherlands': 'city', 'Dubai, United Arab Emirates': 'city',
        'Cancun, Mexico': 'beach', 'Barcelona, Spain': 'beach',
        'Honolulu, Hawaii': 'beach', 'Berlin, Germany': 'city',
        'Marrakech, Morocco': 'city', 'Edinburgh, Scotland': 'city',
        'Rome': 'city', 'Bangkok': 'city', 'Hawaii': 'beach', 'Athens, Greece': 'mountain',
        'Cape Town, South Africa': 'mountain', 'Auckland, New Zealand': 'mountain'
    }
    df['Destination Type'] = df['Destination'].apply(lambda x: destination_type_map.get(x, 'city'))

    # Feature Engineering
    df['Total cost'] = df['Accommodation cost'] + df['Transportation cost']

    # Destination Aggregation
    destination_profiles = df.groupby('Destination').agg({
        'Destination Type': 'first',
        'Total cost': 'mean',
        'Duration (days)': 'mean',
        'Accommodation type': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
        'Transportation type': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    }).reset_index()

    # Feature Preprocessor Matrix
    num_features = ['Total cost', 'Duration (days)']
    cat_features = ['Destination Type', 'Accommodation type', 'Transportation type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )

    feature_matrix = preprocessor.fit_transform(destination_profiles)

    return destination_profiles, preprocessor, feature_matrix

destination_profiles, preprocessor, feature_matrix = load_and_preprocess_data()

# App UI
st.title("✈️ AI Travel Recommendation System")
st.markdown("Find your next travel destination based on your budget, travel duration, and scenery preference.")

# Sidebar Filters
st.sidebar.header("Your Travel Preferences")
preferred_type = st.sidebar.selectbox("Destination Type", options=["beach", "city", "mountain"])
max_budget = st.sidebar.slider("Maximum Budget ($)", min_value=500, max_value=10000, value=2000, step=100)
preferred_days = st.sidebar.slider("Trip Duration (Days)", min_value=1, max_value=30, value=7)
top_n = st.sidebar.number_input("Top Recommendations", min_value=1, max_value=10, value=3)

if st.button("Get Recommendations"):
    user_input = pd.DataFrame([{
        'Destination Type': preferred_type,
        'Total cost': max_budget,
        'Duration (days)': preferred_days,
        'Accommodation type': 'Unknown',
        'Transportation type': 'Unknown'
    }])

    user_vector = preprocessor.transform(user_input)
    similarity_scores = cosine_similarity(user_vector, feature_matrix).flatten()

    results = destination_profiles.copy()
    results['Match Score (%)'] = (similarity_scores * 100).round(1)
    
    filtered = results[results['Total cost'] <= max_budget * 1.2]
    if filtered.empty:
        filtered = results

    recommendations = filtered.sort_values(by='Match Score (%)', ascending=False).head(top_n)

    st.subheader("Top Recommended Destinations")
    
    cols = st.columns(len(recommendations))
    for idx, (_, row) in enumerate(recommendations.iterrows()):
        with cols[idx]:
            st.metric(label=row['Destination'], value=f"{row['Match Score (%)']}% Match")
            st.write(f"**Type:** {row['Destination Type'].title()}")
            st.write(f"**Est. Cost:** ${row['Total cost']:.2f}")
            st.write(f"**Avg Duration:** {row['Duration (days)']:.0f} days")

    st.dataframe(recommendations[['Destination', 'Destination Type', 'Total cost', 'Duration (days)', 'Match Score (%)']])