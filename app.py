import streamlit as st 
import pandas as pd

import pickle
collegeData=pickle.load(open('D:\CollegeRecommendationSystem\DummyModel\colleges.pkl','rb'))

college_list1=collegeData['College Name'].values

similarity=pickle.load(open('D:\CollegeRecommendationSystem\DummyModel\similarity.pkl','rb'))

# defining function for recommendation

def recommend(college):
    college_index = collegeData[collegeData['College Name'] == college].index[0]
    similar = similarity[college_index]
    
    # Get indices of top 10 most similar colleges (excluding the college itself)
    college_list = sorted(list(enumerate(similar)), reverse=True, key=lambda x: x[1])[1:11]
    
    recommended_colleges=[]
    # Print the recommended colleges
    for i in college_list:
        recommended_colleges.append(collegeData.iloc[i[0]]['College Name'].title()+", "+collegeData.iloc[i[0]]['City']+", "+collegeData.iloc[i[0]]['State'])
        
    return recommended_colleges

st.title('College Recommendation System')
selected_College_name=st.selectbox(
    'Search for the College',
    (college_list1)
)

if st.button('Recommend'):
    recommendations=recommend(selected_College_name)
    st.write("the 10 recommended colleges are....")
    for i in recommendations:
        st.success(i)