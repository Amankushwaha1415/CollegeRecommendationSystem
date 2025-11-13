import streamlit as st
import pickle
import re
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity


#  Load Data and Models

df = pickle.load(open('data.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
Vectors_model = pickle.load(open('Vectors_model.pkl', 'rb'))

df['Facilities'] = df['Facilities'].apply(lambda lst: [f.replace("Medical/Hospital", "Medical") for f in lst])

collegeData = pickle.load(open('D:\\CollegeRecommendationSystem\\DummyModel\\colleges.pkl', 'rb'))
similarity = pickle.load(open('D:\\CollegeRecommendationSystem\\DummyModel\\similarity.pkl', 'rb'))
college_list1 = collegeData['College Name'].values


#  Custom Styling

st.set_page_config(page_title="College Recommendation System", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #f5f6fa;
        }
        .main-title {
            text-align:center;
            font-size:42px !important;
            font-weight:800;
            color:#000000;
            margin-top: 10px;
        }
        .subtitle {
            text-align:center;
            font-size:20px;
            color:#424242;
            margin-bottom:40px;
        }
        .stButton>button {
            background-color: #3949ab;
            color: white;
            border: none;
            padding: 0.6em 1.2em;
            border-radius: 12px;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #283593;
            transform: scale(1.02);
        }
        .stRadio > label {
            font-weight: bold;
        }
        .stSelectbox, .stMultiSelect, .stTextInput {
            background-color: #ffffff;
            border-radius: 8px !important;
        }
        h1, h2, h3, h4 {
            color: #000000;
        }
        .css-1v0mbdj {
            background-color: #fafafa !important;
            border-radius: 12px !important;
            padding: 20px !important;
        }
        .stSelectbox label,
        .stMultiSelect label,
        .stRadio label,
        .stTextInput label,
        .stSlider label {
            color: #000000 !important;       /* black */
            font-weight: 900 !important;     /* extra bold */
            font-size: 18px !important;      /* slightly bigger */
}
    </style>
""", unsafe_allow_html=True)


#  Helper Functions

def unique_elements_from_column(df_column):
    unique_elements = set()
    for lst in df_column:
        unique_elements.update(lst)
    return list(unique_elements)

unique_states = list(unique_elements_from_column(df['State2']))

def find_location(data):
    dict_city_state = {}
    for state in unique_states:
        cities = set()
        data_city = data[data['State2'].apply(lambda x: x[0] == state)]
        for _, row in data_city.iterrows():
            cities.update(row['City2'])
        dict_city_state[state] = list(cities)
    return dict_city_state

#  Categorize Courses (Smart Integrated Detection)

def categorize_courses(course_list):
    def match_patterns(course, patterns):
        course_lower = course.lower()
        return any(re.search(p, course_lower) for p in patterns)

    patterns = {
        "btech_be": [r"\bb[\.\s]*tech\b", r"\bb[\.\s]*e\b", r"bachelor\s+of\s+technology", r"bachelor\s+of\s+engineering"],
        "mtech_me": [r"\bm[\.\s]*tech\b", r"\bm[\.\s]*e\b", r"master\s+of\s+technology", r"master\s+of\s+engineering"],
        "bsc": [r"\bb[\.\s]*sc\b", r"bachelor\s+of\s+science"],
        "msc": [r"\bm[\.\s]*sc\b", r"master\s+of\s+science"],
        "bca_mca": [r"\bbca\b", r"\bmca\b", r"bachelor\s+of\s+computer\s+applications", r"master\s+of\s+computer\s+applications"],
        "phd": [r"\bph\.?d\b", r"doctor\s+of\s+philosophy"],
        "diploma": [r"\bdiploma\b", r"\bpolytechnic\b", r"pg\s*diploma"]
    }

    integrated, btech_be, mtech_me, bsc, msc, bca_mca, phd, diploma = [], [], [], [], [], [], [], []

    for course in course_list:
        cl = course.lower()
        has_btech = any(re.search(p, cl) for p in patterns["btech_be"])
        has_mtech = any(re.search(p, cl) for p in patterns["mtech_me"])

        if ("integrated" in cl and (has_btech or has_mtech)) or (has_btech and has_mtech):
            integrated.append(course)
        elif has_btech:
            btech_be.append(course)
        elif has_mtech:
            mtech_me.append(course)
        elif match_patterns(course, patterns["bsc"]):
            bsc.append(course)
        elif match_patterns(course, patterns["msc"]):
            msc.append(course)
        elif match_patterns(course, patterns["bca_mca"]):
            bca_mca.append(course)
        elif match_patterns(course, patterns["phd"]):
            phd.append(course)
        elif match_patterns(course, patterns["diploma"]):
            diploma.append(course)

    all_categorized = set(btech_be + mtech_me + integrated + bsc + msc + bca_mca + phd + diploma)
    others = [c for c in course_list if c not in all_categorized]

    return btech_be, mtech_me, integrated, bsc, msc, bca_mca, phd, diploma, others


#  Prepare Data

uniqueCourses = list(unique_elements_from_column(df['Courses']))
unique_facilities = list(unique_elements_from_column(df['Facilities']))
unique_city_state_dict = find_location(df)

fees_options = ['High Fees', 'Medium Fees', 'Low Fees']
college_type = ['Private', 'Public']
establishment_type = ['New College', 'Old College']
gender_accepted = ['Co-Ed', 'Female', 'Male']

btech_be, mtech_me, integrated, bsc, msc, bca_mca, phd, diploma, others = categorize_courses(uniqueCourses)


# Header Section

st.markdown("<div class='main-title'>College Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Find the best colleges based on your preferences or explore similar colleges easily.</div>", unsafe_allow_html=True)

# Sidebar Mode Selector

mode = st.sidebar.radio(
    "Choose Recommendation Mode",
    ["Preference-Based Recommendation", "College Name-Based Recommendation"]
)

# Backend API
BACKEND_URL = "http://127.0.0.1:8000"


# MODE 1: Preference-Based Recommendation

if mode == "Preference-Based Recommendation":
    st.header("Preference-Based Recommendation")
    st.write("Customize your preferences below and get AI-based college suggestions.")

    st.markdown("---")

    # --- 3 columns for course selection ---
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_btech_be = st.multiselect("B.Tech / B.E Courses", sorted(btech_be))
        selected_bsc = st.multiselect("B.Sc Courses", sorted(bsc))
        selected_bca_mca = st.multiselect("BCA / MCA Courses", sorted(bca_mca))
    with col2:
        selected_mtech_me = st.multiselect("M.Tech / M.E Courses", sorted(mtech_me))
        selected_msc = st.multiselect("M.Sc Courses", sorted(msc))
        selected_diploma = st.multiselect("Diploma Courses", sorted(diploma))
    with col3:
        selected_integrated = st.multiselect("Integrated B.Tech + M.Tech Courses", sorted(integrated))
        selected_phd = st.multiselect("Ph.D Courses", sorted(phd))
        selected_others = st.multiselect("Other Courses", sorted(others))
    selected_courses = (
        selected_btech_be + selected_mtech_me + selected_integrated +
        selected_bsc + selected_msc + selected_bca_mca +
        selected_phd + selected_diploma + selected_others
    )

    selected_facilities = st.multiselect("Select Facilities", sorted(unique_facilities))

    st.markdown("---")

    # --- 4 columns for radio buttons ---
    colA, colB, colC, colD = st.columns(4)
    with colA: selected_fees = st.radio("Fee Range", fees_options)
    with colB: selected_college_type = st.radio("College Type", college_type)
    with colC: selected_establishment = st.radio("Establishment Type", establishment_type)
    with colD: selected_gender = st.radio("Gender Accepted", gender_accepted)

    selected_state = st.selectbox("Select State", list(unique_city_state_dict.keys()))

    st.markdown("---")

    if st.button("Recommend Colleges"):
        payload = {
            "courses": selected_courses,
            "facilities": selected_facilities,
            "fees": selected_fees,
            "college_type": selected_college_type,
            "establishment": selected_establishment,
            "gender": selected_gender,
            "state": selected_state
        }

        with st.spinner("Finding best matches..."):
            response = requests.post(f"{BACKEND_URL}/recommend/preferences", json=payload)
            if response.status_code == 200:
                recs = response.json()["recommendations"]
                st.success("Recommendations ready!")
                st.markdown("### Top Recommended Colleges")
                for i in recs:
                    st.info(f"**{i['college_name']}**, {i['city']}, {i['state']}  _(Similarity: {i['similarity']})_")
            else:
                st.error("Unable to get recommendations. Please check backend connection.")


# MODE 2: College Name-Based Recommendation

elif mode == "College Name-Based Recommendation":
    st.header("College Name-Based Recommendation")
    st.write("Select a college below to find others with similar attributes and ranking.")

    st.markdown("---")

    selected_college_name = st.selectbox("Search for a College", college_list1)

    if st.button("Show Similar Colleges"):
        with st.spinner("Finding similar colleges..."):
            response = requests.post(f"{BACKEND_URL}/recommend/college", json={"college_name": selected_college_name})
            if response.status_code == 200:
                recs = response.json()["recommendations"]
                st.markdown("### Top 10 Similar Colleges:")
                for i in recs:
                    st.success(f"{i['college_name']}, {i['city']}, {i['state']}")
            else:
                st.error("Could not connect to backend.")
