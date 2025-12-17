import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

st.set_page_config(
    page_title="AI Impact on Jobs Analysis",
    layout="wide"
)

st.title("🤖 AI Impact on Jobs Analysis & Prediction System")

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
page = st.sidebar.selectbox(
    "Choose a page",
    ["Data Overview", "Exploratory Data Analysis", "Model Performance", "Job Risk Prediction"]
)

# ==================================================
# PAGE 1: DATA OVERVIEW
# ==================================================
if page == "Data Overview":
    st.header("📊 Data Overview")
    st.write("This dataset contains information about various jobs and their potential automation risk by 2030.")

    st.dataframe(df.head(20))
    st.write(f"**Total Records:** {len(df)}")
    st.write(f"**Columns:** {list(df.columns)}")

    st.subheader("Dataset Info")
    buffer = df.info()
    st.text(buffer)

# ==================================================
# PAGE 2: EDA
# ==================================================
elif page == "Exploratory Data Analysis":
    st.header("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Automation Probability")
        fig, ax = plt.subplots()
        ax.hist(df["Automation_Probability_2030"])
        ax.set_xlabel("Automation Probability (2030)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col2:
        st.subheader("Automation Risk Categories")
        fig, ax = plt.subplots()
        df["Risk_Category"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Risk Category")
        ax.set_ylabel("Number of Jobs")
        st.pyplot(fig)

    st.subheader("AI Exposure vs Automation Probability")
    fig, ax = plt.subplots()
    ax.scatter(df["AI_Exposure_Index"], df["Automation_Probability_2030"])
    ax.set_xlabel("AI Exposure Index")
    ax.set_ylabel("Automation Probability 2030")
    st.pyplot(fig)

# ==================================================
# PAGE 3: MODEL PERFORMANCE
# ==================================================
elif page == "Model Performance":
    st.header("🏆 Model Performance")

    st.write("**Model Used:** Random Forest Regressor")
    st.write("**Target:** Automation_Probability_2030")
    st.write("**Training Performance:**")
    st.write("- R² Score ≈ 0.95")
    st.write("- Tuned using GridSearchCV")

# ==================================================
# PAGE 4: JOB RISK PREDICTION
# ==================================================
elif page == "Job Risk Prediction":
    st.header("🔮 Job Risk Prediction")
    st.write("Enter job-related details to predict automation probability.")

    # -------------------------------
    # Encoders (fit on dataset)
    # -------------------------------
    job_encoder = LabelEncoder()
    job_encoder.fit(df["Job_Title"])

    edu_encoder = OrdinalEncoder(
        categories=[["High School", "Bachelor's", "Master's", "PhD"]]
    )
    edu_encoder.fit(df[["Education_Level"]])

    # -------------------------------
    # INPUT FIELDS
    # -------------------------------
    job_title = st.selectbox(
        "Job Title",
        df["Job_Title"].unique().tolist()
    )

    average_salary = st.number_input(
        "Average Salary",
        min_value=0,
        value=50000
    )

    years_experience = st.slider(
        "Years of Experience",
        0.0, 50.0, 5.0, 0.1
    )

    education_level = st.selectbox(
        "Education Level",
        ["High School", "Bachelor's", "Master's", "PhD"]
    )

    ai_exposure_index = st.slider(
        "AI Exposure Index",
        0.0, 1.0, 0.5, 0.01
    )

    tech_growth_factor = st.slider(
        "Tech Growth Factor",
        0.0, 2.0, 1.0, 0.01
    )

    # -------------------------------
    # Risk Category INPUT (IMPORTANT)
    # -------------------------------
    risk_category_label = st.selectbox(
        "Risk Category (Input Feature)",
        ["Low", "Medium", "High"]
    )

    risk_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }
    risk_category_encoded = risk_mapping[risk_category_label]

    # -------------------------------
    # Skills
    # -------------------------------
    st.subheader("Skills (0.0 to 1.0)")
    col1, col2 = st.columns(2)

    with col1:
        skill_1 = st.slider("Skill 1", 0.0, 1.0, 0.5)
        skill_2 = st.slider("Skill 2", 0.0, 1.0, 0.5)
        skill_3 = st.slider("Skill 3", 0.0, 1.0, 0.5)
        skill_4 = st.slider("Skill 4", 0.0, 1.0, 0.5)
        skill_5 = st.slider("Skill 5", 0.0, 1.0, 0.5)

    with col2:
        skill_6 = st.slider("Skill 6", 0.0, 1.0, 0.5)
        skill_7 = st.slider("Skill 7", 0.0, 1.0, 0.5)
        skill_8 = st.slider("Skill 8", 0.0, 1.0, 0.5)
        skill_9 = st.slider("Skill 9", 0.0, 1.0, 0.5)
        skill_10 = st.slider("Skill 10", 0.0, 1.0, 0.5)

    # -------------------------------
    # Encode inputs
    # -------------------------------
    job_title_encoded = job_encoder.transform([job_title])[0]
    education_encoded = edu_encoder.transform([[education_level]])[0][0]

    # -------------------------------
    # FINAL INPUT DATAFRAME
    # -------------------------------
    input_data = pd.DataFrame([{
        "Job_Title": job_title_encoded,
        "Average_Salary": average_salary,
        "Years_Experience": years_experience,
        "Education_Level": education_encoded,
        "AI_Exposure_Index": ai_exposure_index,
        "Tech_Growth_Factor": tech_growth_factor,
        "Risk_Category": risk_category_encoded,   # ✅ REQUIRED
        "Skill_1": skill_1,
        "Skill_2": skill_2,
        "Skill_3": skill_3,
        "Skill_4": skill_4,
        "Skill_5": skill_5,
        "Skill_6": skill_6,
        "Skill_7": skill_7,
        "Skill_8": skill_8,
        "Skill_9": skill_9,
        "Skill_10": skill_10
    }])

    # -------------------------------
    # PREDICTION
    # -------------------------------
    if st.button("🔮 Predict Automation Probability"):
        prediction = model.predict(input_data)
        prob = prediction[0]

        if prob < 0.3:
            risk = "Low Risk"
            color = "green"
        elif prob < 0.7:
            risk = "Medium Risk"
            color = "orange"
        else:
            risk = "High Risk"
            color = "red"

        st.success(f"Predicted Automation Probability: **{prob:.3f}**")
        # st.mar