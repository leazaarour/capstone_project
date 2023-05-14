import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# App 1

def load_data():
    df = pd.read_csv('/Users/leazaarour/Desktop/data.csv')

    df = df.applymap(lambda x: 1 if x is True else 0 if x is False else x)
    df['disable_person_hh'] = df['disable_person_hh'].apply(lambda x: 1 if x == 'Atleast_one_disable_person' else 0 if x == 'No_disable_person' else x)
    df['i_hh_size_class'] = df['i_hh_size_class'].apply(lambda x: 1 if x == '1 to 4' else 2 if x == '5 to 9' else 3)
    df['i_fcs_category'] = df['i_fcs_category'].apply(lambda x: 1 if x == 'Borderline' else 2 if x == 'Poor' else 3)
    df['i_rcsi_cat'] = df['i_rcsi_cat'].apply(lambda x: 1 if x == 'Low' else 2 if x == 'Medium' else 3)
    df['i_gender_hoh'] = df['i_gender_hoh'].apply(lambda x: 1 if x == 'male' else 0)
    df['i_gender_hoh_co_head'] = df['i_gender_hoh_co_head'].apply(lambda x: 1 if x == 'male' else 0)
    df['i_hoh_age_class'] = df['i_hoh_age_class'].apply(lambda x: 1 if x == 'hoh_age_grater_than_60' else 0)
    df = df.drop(columns=['prefered_modality', 'priority_needs'])

    data_encoded = pd.get_dummies(df)
    df = data_encoded

    scaler = MinMaxScaler()
    df['survey_weight'] = 1 - scaler.fit_transform(df[['survey_weight']])
    df = df.fillna(df.median())

    return df

# def create_and_train_model(df):
#     X = df.drop(columns=['survey_weight'])
#     y = df['survey_weight']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     models = {
#         'Random Forest Regressor': RandomForestRegressor(random_state=42),
#         'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
#         'XGBoost Regressor': XGBRegressor(random_state=42)
#     }

#     for model_name, model in models.items():
#         model.fit(X_train, y_train)

#         y_pred = model.predict(X_test)

#         mse = mean_squared_error(y_test, y_pred)
#         rmse = mean_squared_error(y_test, y_pred, squared=False)
#         mae = mean_absolute_error(y_test, y_pred)
#         medae = median_absolute_error(y_test, y_pred)

#         print(f"{model_name} - MSE: {mse:.7f}, RMSE: {rmse:.7f}, MAE: {mae:.7f}, MedAE: {medae:.7f}")
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

#         ax1.hist(y_test, bins=30, alpha=0.5, label='Actual', color='blue')
#         ax1.hist(y_pred, bins=30, alpha=0.5, label='Predicted', color='orange')
#         ax1.set_xlabel('Target Value')
#         ax1.set_ylabel('Frequency')
#         ax1.set_title(f'{model_name} - Actual vs. Predicted Histogram')
#         ax1.legend()

#         residuals = y_test - y_pred

#         ax2.bar(np.arange(len(y_test)), residuals, color='blue')
#         ax2.set_xlabel('Index')
#         ax2.set_ylabel('Residuals (Actual - Predicted)')
#         ax2.set_title(f'{model_name} - Residuals Bar Plot')

#         plt.show()
def create_and_train_model(df):
    X = df.drop(columns=['survey_weight'])
    y = df['survey_weight']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
        'XGBoost Regressor': XGBRegressor(random_state=42)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)

        print(f"{model_name} - MSE: {mse:.7f}, RMSE: {rmse:.7f}, MAE: {mae:.7f}, MedAE: {medae:.7f}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.hist(y_test, bins=30, alpha=0.5, label='Actual', color='blue')
        ax1.hist(y_pred, bins=30, alpha=0.5, label='Predicted', color='orange')
        ax1.set_xlabel('Target Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{model_name} - Actual vs. Predicted Histogram')
        ax1.legend()

        residuals = y_test - y_pred

        ax2.bar(np.arange(len(y_test)), residuals, color='blue')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Residuals (Actual - Predicted)')
        ax2.set_title(f'{model_name} - Residuals Bar Plot')

        st.pyplot(fig)  # Use st.pyplot instead of plt.show()

def load_data2():
    # Load data
    df = pd.read_csv('D:/April 2023/Lea/data.csv')

    # Preprocessing steps
    df = df.applymap(lambda x: 1 if x is True else 0 if x is False else x)
    df['disable_person_hh'] = df['disable_person_hh'].apply(lambda x: 1 if x == 'Atleast_one_disable_person' else 0 if x == 'No_disable_person' else x)
    df['i_hh_size_class'] = df['i_hh_size_class'].apply(lambda x: 1 if x == '1 to 4' else 2 if x == '5 to 9' else 3)
    df['i_fcs_category'] = df['i_fcs_category'].apply(lambda x: 1 if x == 'Borderline' else 2 if x == 'Poor' else 3)
    df['i_rcsi_cat'] = df['i_rcsi_cat'].apply(lambda x: 1 if x == 'Low' else 2 if x == 'Medium' else 3)
    df['i_gender_hoh'] = df['i_gender_hoh'].apply(lambda x: 1 if x == 'male' else 0)
    df['i_gender_hoh_co_head'] = df['i_gender_hoh_co_head'].apply(lambda x: 1 if x == 'male' else 0)
    df['i_hoh_age_class'] = df['i_hoh_age_class'].apply(lambda x: 1 if x == 'hoh_age_grater_than_60' else 0)
    df = df.drop(columns=['prefered_modality', 'priority_needs'])

    # One-hot encoding
    data_encoded = pd.get_dummies(df)
    df = data_encoded
    df = df[['disable_person_hh', 'i_hh_size_class', 'i_hoh_age_class', 'i_fcs_category', 'i_rcsi_cat', 'i_gender_hoh_co_head', 'total_water_needs', 'i_rcsi', 'i_crowdedness', 'i_prop_communication','survey_weight']]
    df = df.rename(columns={
    'disable_person_hh': 'Disabled_Person_Presence',
    'i_hh_size_class': 'Household_Size_Class',
    'i_hoh_age_class': 'Head_of_Household_Age_Class',
    'i_fcs_category': 'Food_Consumption_Score_Category',
    'i_rcsi_cat': 'Coping_Strategy_Index_Category',
    'i_gender_hoh_co_head': 'Gender_of_Head_of_Household_and_Co_Head',
    'total_water_needs': 'Total_Water_Needs',
    'i_rcsi': 'Reduced_Coping_Strategy_Index',
    'i_crowdedness': 'Household_Crowdedness',
    'i_prop_communication': 'Proportion_of_Communication_Expenses',
    'survey_weight': 'Survey_Weight'
    })

    # Normalize the 'survey_weight' column
    scaler = MinMaxScaler()
    df['Survey_Weight'] = 1 - scaler.fit_transform(df[['Survey_Weight']])
    df = df.fillna(df.median())

    return df

def create_and_train_model2(df):
    # Create and train the model
    X = df.drop(columns=['Survey_Weight'])
    y = df['Survey_Weight']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model
def app1():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.title("Data Analysis App")

    st.write("Please upload your data CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = StringIO(uploaded_file.getvalue().decode('utf-8'))
        df = pd.read_csv(data)
        st.write("Data preview:")
        st.write(df.head())

        df_processed = load_data()
        st.write("Data after processing:")
        st.write(df_processed.head())

        create_and_train_model(df_processed)

        st.write("The above plots compare the prediction with actual values using alot of variables. However to make the prediction page more useful, we performed feature selection and reduced the variables at page 2.")
        # Display the model evaluation results

        # Display the plots

def app2():
    st.title("Model Prediction App")
    df = load_data2()
    # Create the model
    model = create_and_train_model2(df)
 
    # App 2 code...
    # Get user input
    st.subheader("Please input values for prediction")
    Disabled_Person_Presence = st.number_input("Presence of disabled person in family,(1 if disable exist, otherwise 0)")
    Household_Size_Class = st.number_input("Household Size Class,(1 if 1-4, 2 if 5-9, 3 if more than 9")
    Head_of_Household_Age_Class = st.number_input("Age of Head of Household")
    Food_Consumption_Score_Category = st.number_input("Food Consumption Score Category,( 1 if Borderline, 2 if Poor, 3 if acceptable)")
    Coping_Strategy_Index_Category = st.number_input("Coping Strategy Index Category")
    Total_Water_Needs = st.number_input("Total Water Needs")
    Reduced_Coping_Strategy_Index = st.number_input("Reduced Coping Strategies Index,(1 if Low, 2 if Medium, 3 if High)")
    Household_Crowdedness = st.number_input("Household Crowdedness")
    Proportion_of_Communication_Expenses = st.number_input("Proportion of Communication Expenses")
    Gender_of_Head_of_Household_and_Co_Head = st.number_input("Gender of Head of Household and Co-Head, 1 if male, 0 if female")

    user_input = pd.DataFrame([[Disabled_Person_Presence, 
                                Household_Size_Class, 
                                Head_of_Household_Age_Class, 
                                Food_Consumption_Score_Category, 
                                Coping_Strategy_Index_Category, 
                                Gender_of_Head_of_Household_and_Co_Head,
                                Total_Water_Needs, 
                                Reduced_Coping_Strategy_Index, 
                                Household_Crowdedness, 
                                Proportion_of_Communication_Expenses]], 
                              columns=['Disabled_Person_Presence', 
                                       'Household_Size_Class', 
                                       'Head_of_Household_Age_Class', 
                                       'Food_Consumption_Score_Category', 
                                       'Coping_Strategy_Index_Category', 
                                       'Gender_of_Head_of_Household_and_Co_Head',
                                       'Total_Water_Needs', 
                                       'Reduced_Coping_Strategy_Index', 
                                       'Household_Crowdedness', 
                                       'Proportion_of_Communication_Expenses'])

    if st.button('Predict'):
        prediction = model.predict(user_input)
        st.write("The prediction is ", prediction)

def main():
    # Add navigation to switch between apps
    app_options = ["Data Analysis", "Predictions"]
    app_choice = st.sidebar.selectbox("Select App", app_options)

    if app_choice == "Data Analysis":
        app1()
    elif app_choice == "Predictions":
        app2()

if __name__ == "__main__":
    main()

