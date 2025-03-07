import streamlit as st
from PIL import Image
import google.generativeai as genai
from phi.agent import Agent
from phi.model.google import Gemini

# Configure API Keys
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Page Setup
st.set_page_config(page_title="Pet Wellness Assistant", page_icon="🐾", layout="centered")
st.title("Pet Wellness Assistant")
st.markdown("Helping you understand your pet's well-being 🐶🐱")

def analyze_pet_image(image):
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content([
        "Analyze this pet photo for general wellness assessment. Focus on:",
        "- Possible breed identification",
        "- Visible health indicators",
        "- Estimated weight",
        "- Coat and skin condition",
        "Format: Simple and user-friendly insights",
        image
    ])
    return response.text

def generate_recommendations(agent, analysis, details):
    prompt = f"""
    **Pet Wellness Check Summary**:
    {analysis}

    **Owner's Input**:
    {details}

    Provide:
    1. Basic care recommendations
    2. Nutrition suggestions based on age
    3. Common health signs to monitor
    4. When professional veterinary care is needed
    """
    return agent.run(prompt).content

# Initialize AI
pet_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions="Provide clear, practical pet wellness advice. Keep it simple and actionable."
)

# Image Upload
uploaded_image = st.file_uploader("Upload a clear photo of your pet", type=["jpg", "jpeg", "png"])

# User Input
with st.form("pet_info"):
    age = st.number_input("Pet's age (months)", min_value=1, max_value=360, value=6)
    concern = st.selectbox(
        "What's your primary concern?",
        ("Routine checkup", "Skin & coat", "Digestion", "Behavior", "Weight management")
    )
    diet = st.text_input("Current diet", placeholder="Brand/type of food")
    
    submit = st.form_submit_button("Get Wellness Report")

if submit:
    if not uploaded_image:
        st.warning("Please upload a photo of your pet.")
    else:
        with st.spinner("Analyzing your pet's wellness..."):
            try:
                # Process image
                img = Image.open(uploaded_image)
                analysis = analyze_pet_image(img)
                
                # User details
                details = f"""
                - Age: {age} months
                - Concern: {concern}
                - Current Diet: {diet}
                """
                
                # Generate recommendations
                st.subheader("Pet Wellness Overview")
                st.write(analysis)
                
                recommendations = generate_recommendations(pet_agent, analysis, details)
                st.subheader("Personalized Care Advice")
                st.markdown(recommendations)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Note: This tool provides general wellness insights. Consult a vet for professional medical advice.")
