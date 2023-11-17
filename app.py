import streamlit as st
import pickle
import pandas as pd
from keras.models import load_model

# Helper objects for scaling and label encoding
with open('label_encoder_1.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('scaler_1.pkl', 'rb') as file:
    sc = pickle.load(file)

# Read model object from model file
loaded_model = load_model('churn_class_3.h5')


# Function to perform some action with input values
def process_inputs(tenure, gender, s_citizen, partner, dependents, 
                   total_charges, phone_service, m_lines, i_service, o_security, 
                   monthly_charges,o_backup, d_protection, t_support, streaming_tv, 
                   streaming_movies, contract, paperless_billing, payment_method):

    input_data = [gender, partner, dependents, phone_service, 
                  m_lines, i_service, o_security, o_backup, d_protection, 
                  t_support, streaming_tv, streaming_movies, contract, 
                  paperless_billing, payment_method, s_citizen, tenure, 
                  monthly_charges, total_charges
                ]

    for key, value in enumerate(input_data):
        if (key != 16 and key != 17 and key != 18 ):
            input_data[key] = encoder.fit_transform([value])
    
    scaled_input = sc.transform([input_data])

    return scaled_input

# Provie list of options required in the input fields
gender_opt = ['Male', 'Female']
multiple_lines_opt = ['No phone service', 'No', 'Yes']
internet_service_opt = ['DSL', 'Fiber optic']
contract_opt = ['Month-to-month', 'One year', 'Two year']
payment_meth_opt = ['Electronic check', 'Mailed check', 'Bank transfer(automatic)', 'Credit card(automatic)']
yes_or_not_opt = ['Yes', 'No']

# Streamlit app
def main():
    # specify title and style of the title
    st.markdown("<h1 style='text-align: center; color: orange;'>Predict likelihood of churn</h1>", unsafe_allow_html=True)

    # Create 4 columns
    col1, col2, col3, col4 = st.columns(4)

    # Structure the input boxes for the fields that will be taking input
    with col1:
        tenure = st.number_input("Tenure", value=0)
        gender = st.selectbox('Gender', gender_opt)
        s_citizen = st.selectbox('Senior Citizen', yes_or_not_opt)
        partner = st.selectbox('partner', yes_or_not_opt)
        dependents = st.selectbox('Dependents', yes_or_not_opt)

    with col2:
        total_charges = st.number_input("Total Charges", value=0.00)
        phone_service = st.selectbox('Phone service', yes_or_not_opt)
        m_lines = st.selectbox('Multiple Lines', multiple_lines_opt)
        i_service = st.selectbox('Internet Service', internet_service_opt)
        o_security = st.selectbox('Online Security', yes_or_not_opt)

    with col3:
        monthly_charges = st.number_input("Monthly Charges", value=0.00)
        o_backup = st.selectbox('Online Backup', yes_or_not_opt)
        d_protection = st.selectbox('Device Protection', yes_or_not_opt)
        t_support = st.selectbox('Tech Support', yes_or_not_opt)
        streaming_tv = st.selectbox('Streaming TV', yes_or_not_opt)

    with col4:
        streaming_movies = st.selectbox('Streaming Movies', yes_or_not_opt)
        contract = st.selectbox('Contract', contract_opt)
        paperless_billing = st.selectbox('Paperless Billing', yes_or_not_opt)
        payment_method = st.selectbox('Payment Method', payment_meth_opt)

    # Submit button
    if st.button("Submit"):
        # Call the function and pass input values
        args = process_inputs(tenure, gender, s_citizen, partner, dependents, 
                   total_charges, phone_service, m_lines, i_service, o_security, 
                   monthly_charges,o_backup, d_protection, t_support, streaming_tv, 
                   streaming_movies, contract, paperless_billing, payment_method)
        
        predicted = loaded_model.predict(args)

        # Store multiple predictions
        # preds = []

        # # Put together list of predictions for confidence calculation
        # for i in range(5):
        #     predicted = loaded_model.predict(args)
        #     preds.append(predicted)

        # # Calculate confidence of prediction
        # lower_bound = np.percentile(preds, 2.5)
        # upper_bound = np.percentile(preds, 97.5)

        # alpha = upper_bound - lower_bound
        # confidence = (1 - alpha) * 100

        
        # Display the result
        pred_value = round(predicted[0,0] * 100, 2)
        st.success(f"I am {pred_value}% confident that this user will churn.")
        # st.success(f"This model is {confidence}% confident.")

if __name__ == "__main__":
    main()
