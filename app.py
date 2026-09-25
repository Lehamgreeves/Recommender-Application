import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Travel Recommender Dual Engine", page_icon="✈️", layout="wide")

# ==========================================
# 1. DATA PREPROCESSING & PIPELINE
# ==========================================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('traveler-trip-data.csv')

    # Data Cleaning
    df['Destination'] = df['Destination'].fillna('Unknown')
    df['Traveler name'] = df['Traveler name'].fillna('Unknown')
    df['Traveler age'] = df['Traveler age'].fillna(df['Traveler age'].median())
    df['Accommodation cost'] = pd.to_numeric(df['Accommodation cost'], errors='coerce').fillna(0)
    df['Transportation cost'] = pd.to_numeric(df['Transportation cost'], errors='coerce').fillna(0)
    df['Duration (days)'] = df['Duration (days)'].fillna(df['Duration (days)'].median())
    df['Accommodation type'] = df['Accommodation type'].fillna('Unknown')
    df['Transportation type'] = df['Transportation type'].fillna('Unknown')

    # Map Destination Categories
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
    df['Total cost'] = df['Accommodation cost'] + df['Transportation cost']

    return df

df = load_and_clean_data()

# ==========================================
# 2. CONTENT-BASED FILTERING SETUP
# ==========================================
@st.cache_data
def build_content_model(df):
    destination_profiles = df.groupby('Destination').agg({
        'Destination Type': 'first',
        'Total cost': 'mean',
        'Duration (days)': 'mean',
        'Accommodation type': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
        'Transportation type': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
    }).reset_index()

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

destination_profiles, preprocessor, content_feature_matrix = build_content_model(df)

# ==========================================
# 3. COLLABORATIVE FILTERING SETUP
# ==========================================
@st.cache_data
def build_collaborative_model(df):
    df_collab = df.copy()
    df_collab['Cost_Per_Day'] = df_collab['Total cost'] / np.maximum(df_collab['Duration (days)'], 1)
    
    min_cpd = df_collab['Cost_Per_Day'].min()
    max_cpd = df_collab['Cost_Per_Day'].max()
    df_collab['Implicit_Rating'] = 1 + 4 * (df_collab['Cost_Per_Day'] - min_cpd) / (max_cpd - min_cpd + 1e-5)
    
    user_item_matrix = df_collab.pivot_table(
        index='Traveler name', 
        columns='Destination', 
        values='Implicit_Rating', 
        aggfunc='mean'
    )
    
    user_item_filled = user_item_matrix.fillna(0)
    user_similarity = cosine_similarity(user_item_filled)
    user_similarity_df = pd.DataFrame(
        user_similarity, 
        index=user_item_matrix.index, 
        columns=user_item_matrix.index
    )
    
    return user_item_matrix, user_similarity_df

user_item_matrix, user_similarity_df = build_collaborative_model(df)

# ==========================================
# 4. APP INTERFACE WITH SEPARATE TABS
# ==========================================
st.title("✈️ Travel Recommendation Engine")
st.markdown("Comparing Content-Based and Collaborative Filtering using `traveler-trip-data.csv`.")

tab1, tab2 = st.tabs(["🧩 Content-Based Algorithm", "🤝 Collaborative Filtering Algorithm"])

# ------------------------------------------
# TAB 1: CONTENT-BASED ALGORITHM
# ------------------------------------------
with tab1:
    st.header("Content-Based Recommendation")
    st.caption("Recommends destinations by calculating feature similarity (Cost, Duration, Scenery, Accommodation).")
    
    mode = st.radio(
        "Recommendation Mode:", 
        ["User Preference Matching", "Similar Destination Search"], 
        key="cb_mode_radio"
    )
    
    if mode == "User Preference Matching":
        c1, c2, c3 = st.columns(3)
        pref_type = c1.selectbox("Preferred Scenery", options=["beach", "city", "mountain"], key="cb_pref_type_select")
        max_budget = c2.slider("Max Budget ($)", 500, 10000, 2000, 100, key="cb_budget_slider")
        pref_days = c3.slider("Trip Duration (Days)", 1, 30, 7, key="cb_days_slider")
        
        if st.button("Find Matching Destinations", key="cb_btn_matching"):
            user_input = pd.DataFrame([{
                'Destination Type': pref_type,
                'Total cost': max_budget,
                'Duration (days)': pref_days,
                'Accommodation type': 'Unknown',
                'Transportation type': 'Unknown'
            }])
            
            user_vector = preprocessor.transform(user_input)
            sim_scores = cosine_similarity(user_vector, content_feature_matrix).flatten()
            
            res = destination_profiles.copy()
            res['Match Score (%)'] = (sim_scores * 100).round(1)
            filtered = res[res['Total cost'] <= max_budget * 1.2]
            if filtered.empty:
                filtered = res
            
            top_recs = filtered.sort_values(by='Match Score (%)', ascending=False).head(3)
            
            cols = st.columns(len(top_recs))
            for idx, (_, row) in enumerate(top_recs.iterrows()):
                with cols[idx]:
                    st.metric(row['Destination'], f"{row['Match Score (%)']}% Match")
                    st.write(f"**Type:** {row['Destination Type'].title()}")
                    st.write(f"**Est. Cost:** ${row['Total cost']:.2f}")

    else:
        target_dest = st.selectbox(
            "Select Target Destination:", 
            options=destination_profiles['Destination'].unique(), 
            key="cb_target_dest_select"
        )
        if st.button("Find Similar Destinations", key="cb_btn_similar"):
            idx = destination_profiles[destination_profiles['Destination'] == target_dest].index[0]
            sim_scores = cosine_similarity(content_feature_matrix[idx], content_feature_matrix).flatten()
            
            similar_indices = sim_scores.argsort()[::-1]
            similar_indices = [i for i in similar_indices if i != idx][:3]
            
            recs = destination_profiles.iloc[similar_indices].copy()
            recs['Similarity Score (%)'] = (sim_scores[similar_indices] * 100).round(1)
            
            st.dataframe(recs[['Destination', 'Destination Type', 'Total cost', 'Duration (days)', 'Similarity Score (%)']])

# ------------------------------------------
# TAB 2: COLLABORATIVE FILTERING ALGORITHM
# ------------------------------------------
with tab2:
    st.header("Collaborative Filtering Recommendation")
    st.caption("Recommends unvisited destinations by analyzing user-user behavioral similarities from past trips.")
    
    selected_traveler = st.selectbox(
        "Select Traveler Name:", 
        options=user_item_matrix.index, 
        key="cf_traveler_select"
    )
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader(f"Travel History for {selected_traveler}")
        history = df[df['Traveler name'] == selected_traveler][['Destination', 'Duration (days)', 'Total cost', 'Accommodation type']]
        st.dataframe(history, use_container_width=True)
        
        if st.button("Get Collaborative Recommendations", key="cf_btn_recs"):
            similar_users = user_similarity_df[selected_traveler].sort_values(ascending=False).drop(selected_traveler)
            user_ratings = user_item_matrix.loc[selected_traveler]
            unvisited = user_ratings[user_ratings.isna()].index
            
            predicted_ratings = {}
            for dest in unvisited:
                other_ratings = user_item_matrix[dest].dropna()
                common = similar_users.index.intersection(other_ratings.index)
                
                if len(common) > 0:
                    sims = similar_users.loc[common]
                    rats = other_ratings.loc[common]
                    if sims.sum() > 0:
                        pred = np.dot(sims, rats) / sims.sum()
                        predicted_ratings[dest] = round(pred, 2)
            
            rec_df = pd.DataFrame(list(predicted_ratings.items()), columns=['Destination', 'Predicted Affinity Score'])
            rec_df = rec_df.sort_values(by='Predicted Affinity Score', ascending=False).head(3)
            
            st.subheader("🎯 Recommended for You (Based on Similar Travelers)")
            if not rec_df.empty:
                cols = st.columns(len(rec_df))
                for idx, (_, row) in enumerate(rec_df.iterrows()):
                    with cols[idx]:
                        st.metric(row['Destination'], f"{row['Predicted Affinity Score']} / 5.0")
            else:
                st.info("No unvisited destinations with sufficient peer coverage found.")

    with col_right:
        st.subheader("👥 Most Similar Travelers")
        top_peers = user_similarity_df[selected_traveler].sort_values(ascending=False).drop(selected_traveler).head(3)
        for peer_name, sim_score in top_peers.items():
            peer_info = df[df['Traveler name'] == peer_name].iloc[0]
            st.write(f"**{peer_name}**")
            st.caption(f"Similarity: `{sim_score:.2f}` | Nationality: {peer_info['Traveler nationality']}")
            st.divider()