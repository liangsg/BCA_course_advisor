# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
# from helper_functions import llm
from helper_functions.utility import check_password
from logics.customer_query_handler import custom_qa_with_rerank

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("Streamlit App")

if not check_password():
    st.stop()

form = st.form(key="form")
form.subheader("Prompt")

user_prompt = form.text_area("Enter your prompt here", height=200)

if form.form_submit_button("Submit"):
    st.toast(f"User Input Submitted - {user_prompt}")

    st.divider()

#    response, course_details = process_user_message(user_prompt)
#    st.write(response)
    response = custom_qa_with_rerank(user_prompt,top_k_retrieve=3)
    st.divider()
    print(type(response))
    st.write(response['result']['text'])
    #    print(course_details)
    #  df = pd.DataFrame(course_details)
    #  df
