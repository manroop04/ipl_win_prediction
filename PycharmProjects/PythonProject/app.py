import pandas as pd
import streamlit as st
import pickle

teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select the batting team", sorted(teams))
with col2:
    bowling_team = st.selectbox("Select the bowling team", sorted(teams))

selected_city = st.selectbox('Select Host City', sorted(cities))
target = st.number_input('Target', min_value=0, value=0)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, value=0)

with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, value=0.0)

with col5:
    wickets = st.number_input('Wickets Out', min_value=0, max_value=10, value=0)

# Initialize variables
runs_left = 0
balls_left = 0
wickets_left = 0
crr = 0
rrr = 0

if st.button('Predict Probability'):
    if overs > 0:
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        st.write("Runs Left:", runs_left)
        st.write("Balls Left:", balls_left)
        st.write("Wickets Left:", wickets_left)
        st.write("Current Run Rate (CRR):", crr)
        st.write("Required Run Rate (RRR):", rrr)
    else:
        st.error("Overs completed cannot be zero.")

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    st.table(input_df)

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team+'- '+str(round(win*100))+'%')
    st.header(bowling_team + '- ' + str(round(loss * 100)) + '%')


