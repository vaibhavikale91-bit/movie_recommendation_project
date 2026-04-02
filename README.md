# movie_recommendation_project
🎬 Smart Movie Recommender using Machine Learning and Streamlit for personalized movie suggestions
# 🎬 Movie Recommendation System

## 📌 Project Overview
This project is an advanced Movie Recommendation System built using Python, SQL concepts, and Machine Learning techniques.  
It recommends movies based on user preferences using a hybrid approach.

---

## 🚀 Features
✔ Content-Based Filtering (based on movie genres)  
✔ Collaborative Filtering (based on user ratings)  
✔ Hybrid Recommendation System (combined approach)  
✔ Interactive UI using Streamlit  
✔ Movie details (Genre, Rating)  
✔ Data Visualization (Ratings distribution)

---

## 🧠 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

## 📂 Project Structure
movie_recommendation_project/
│── app.py
│── data/
│ ├── movies.csv
│ ├── ratings.csv
│── models/
│ ├── content_model.py
│ ├── collab_model.py
│ ├── hybrid_model.py
│── venv/
│── README.md


---

## ⚙️ How It Works

### 1. Content-Based Filtering
- Uses TF-IDF on movie genres  
- Finds similar movies  

### 2. Collaborative Filtering
- Uses user ratings  
- Finds similar user behavior  

### 3. Hybrid Model
- Combines both approaches  
- Uses weighted scoring for better accuracy  

---

## ▶️ How to Run the Project

### Step 1: Activate Virtual Environment

venv\Scripts\activate


### Step 2: Install Requirements

pip install pandas numpy scikit-learn streamlit matplotlib


### Step 3: Run Application

streamlit run app.py

### step 4: In App.py add your API Key
 api_key = "YOUR_API_KEY"  # Replace with your actual TMDB API key
---
## 🔑 Get TMDB API Key (Short Steps)
Go to 👉 https://www.themoviedb.org/
Sign up & login
Open 👉 Settings → API
Click Create API
Copy your key

## Show in UI
poster = fetch_poster(movie)
st.image(poster, width=150)

## 📊 Sample Output
- Select a movie  
- Get top recommended movies  
- View genre and rating  

---

## 💼 Use Cases
- Movie streaming platforms  
- Recommendation systems  
- Data analytics projects  

---

## 🧠 Learning Outcomes
- Data preprocessing and EDA  
- Machine Learning models  
- Recommendation systems  

## OutPut
![Alt text](image.png)
![Alt text](image-1.png)
![Alt text](image-2.png)
![Alt text](image-3.png)
