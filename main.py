from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI(title="College Recommendation Backend API")

# ---------------- Load Models & Data ----------------
df = pickle.load(open('data.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
Vectors_model = pickle.load(open('Vectors_model.pkl', 'rb'))
collegeData = pickle.load(open('colleges.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# ---------------- Request Models ----------------
class PreferenceRequest(BaseModel):
    courses: list
    facilities: list
    fees: str
    college_type: str
    establishment: str
    gender: str
    state: str

class CollegeNameRequest(BaseModel):
    college_name: str

# ---------------- Helper Function ----------------
def clean_and_join(lst):
    return " ".join(item.replace(" ", "").lower() for item in lst)

# ---------------- Preference-Based Recommendation ----------------
@app.post("/recommend/preferences")
def recommend_preferences(data: PreferenceRequest):
    try:
        user_input_string = f"{clean_and_join(data.courses)} {clean_and_join(data.facilities)} " \
                            f"{data.fees} {data.college_type} {data.establishment} {data.gender} {data.state}".strip().lower()

        input_vector = tfidf.transform([user_input_string])
        similarity_scores = cosine_similarity(input_vector, Vectors_model).flatten()
        top_indices = sorted(list(enumerate(similarity_scores)), reverse=True, key=lambda x: x[1])[0:20]

        results = [
            {
                "college_name": df.iloc[i[0]]['College Name'],
                "city": df.iloc[i[0]]['City'],
                "state": df.iloc[i[0]]['State'],
                "similarity": round(i[1], 3)
            }
            for i in top_indices
        ]
        return {"recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- College-Name-Based Recommendation ----------------
@app.post("/recommend/college")
def recommend_college(data: CollegeNameRequest):
    try:
        college = data.college_name
        if college not in collegeData['College Name'].values:
            raise HTTPException(status_code=404, detail="College not found in database")

        index = collegeData[collegeData['College Name'] == college].index[0]
        similar = similarity[index]
        similar_list = sorted(list(enumerate(similar)), reverse=True, key=lambda x: x[1])[1:11]

        results = [
            {
                "college_name": collegeData.iloc[i[0]]['College Name'].title(),
                "city": collegeData.iloc[i[0]]['City'],
                "state": collegeData.iloc[i[0]]['State']
            }
            for i in similar_list
        ]
        return {"recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run: uvicorn main:app --reload

