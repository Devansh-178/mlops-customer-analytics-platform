# dashboard/app.py

import streamlit as st
from api_client import predict, APIUnreachableError, ValidationError, ServerError

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

st.title("Customer Churn Prediction")
st.caption("Single-customer prediction — calls the FastAPI /predict endpoint")

# --- These two live OUTSIDE the form on purpose ---
# They gate which options are valid for MultipleLines / the internet-dependent
# fields below. If they were inside the form, changing them wouldn't trigger
# a rerun, so the dependent fields wouldn't update until after submission.
st.subheader("Services")
phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="phone_service")
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet_service")

with st.form("customer_form"):
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen_label = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col2:
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    st.subheader("Account")
    col3, col4 = st.columns(2)
    with col3:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col4:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    st.subheader("Charges")
    col5, col6 = st.columns(2)
    with col5:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
    with col6:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)

    # --- MultipleLines depends on PhoneService ---
    # A customer with no phone service literally can't have multiple lines.
    # The API's Literal type would accept "Yes" here regardless, but there's
    # no reason the dashboard should ever be able to construct that
    # nonsensical combination in the first place.
    if phone_service == "No":
        multiple_lines = "No phone service"
        st.selectbox("Multiple Lines", ["No phone service"], disabled=True)
    else:
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])

    # --- Internet-dependent fields follow the same logic ---
    internet_dependent_fields = {}
    internet_labels = {
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
    }

    if internet_service == "No":
        st.caption("No internet service selected — dependent fields locked to 'No internet service'")
        for field, label in internet_labels.items():
            st.selectbox(label, ["No internet service"], disabled=True, key=f"disabled_{field}")
            internet_dependent_fields[field] = "No internet service"
    else:
        col7, col8 = st.columns(2)
        for i, (field, label) in enumerate(internet_labels.items()):
            target_col = col7 if i % 2 == 0 else col8
            with target_col:
                internet_dependent_fields[field] = st.selectbox(label, ["Yes", "No"], key=field)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    senior_citizen = 1 if senior_citizen_label == "Yes" else 0

    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": internet_dependent_fields["OnlineSecurity"],
        "OnlineBackup": internet_dependent_fields["OnlineBackup"],
        "DeviceProtection": internet_dependent_fields["DeviceProtection"],
        "TechSupport": internet_dependent_fields["TechSupport"],
        "StreamingTV": internet_dependent_fields["StreamingTV"],
        "StreamingMovies": internet_dependent_fields["StreamingMovies"],
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        result = predict(payload)
    except APIUnreachableError as e:
        st.error(f"Can't reach the prediction API. Is uvicorn running?\n\n`{e}`")
    except ValidationError as e:
        st.error(f"The API rejected this payload (422). This shouldn't happen given the form's constraints — worth investigating.\n\nDetails: {e.detail}")
    except ServerError as e:
        st.error(f"The API hit an internal error (500) while processing this request.\n\nDetails: {e.detail}")
    else:
        prediction = result["churn_prediction"]
        probability = result["churn_probability"]

        st.divider()
        st.subheader("Result")

        if prediction == "Yes":
            st.error("⚠️ Predicted: **Likely to churn**")
        else:
            st.success("✅ Predicted: **Likely to stay**")

        st.metric("Churn Probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))