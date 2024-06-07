import streamlit as st

from streamlit_option_menu import option_menu
import altair as alt

from Predict import args, predict_from_text, predict_from_dataset

alt.themes.enable("dark")

#Code Generator and Q&A form =>
selected = option_menu(None, ["Code Generator", "Question Answering"], 
    icons=['robot', 'chat'], 
    default_index=0, orientation="horizontal",
    styles={
    "container": {"padding": "important", "background-color": "#0F161E", "border-radius" : "15px"},
    "icon": {"color": "white", "font-size": "20px"}, 
    "nav-link": {"font-size": "20px", "text-align": "center", "margin":"0px", "--hover-color": "#eee", "border-radius" : "20px"},
    "nav-link-selected": {"background-color": "#3D89B3"},
    }
)

if selected == "Code Generator":
    st.title("")
    header = st.columns([1,2,1])
    header[0].subheader("")
    header[1].subheader(":blue[Please Explain The Function]")
    header[2].subheader("")
    st.write("")

    Text_Box = st.text_area("Text Box", height=200, placeholder="for example : write function can add tow number")
    Generator = st.button("Generate The Code", use_container_width=True)

    if Generator:
        Code = predict_from_text(args, Text_Box)
        st.write("")
        st.subheader(":blue[Your Function : ]")
        st.write("")
        st.code(Code, language='python', line_numbers=True)

if selected == "Question Answering":
    st.write("Hello")