import streamlit as st
from PIL import Image
import google.generativeai as genai
from phi.agent import Agent
from phi.model.google import Gemini

# Configure API Keys
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Page Setup
st.set_page_config(page_title="Pet Health Assistant", page_icon="🐾", layout="centered")
st.title("Simple Pet Checkup")

def analyze_pet_image(image):
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content([
        "Analyze this pet photo for veterinary assessment. Focus on:",
        "- Breed identification",
        "- Visible health indicators",
        "- Weight estimation",
        "- Coat/skin condition",
        "Format: Plain text observations",
        image
    ])
    return response.text

def generate_report(agent, analysis, details):
    prompt = f"""
    **Pet Photo Analysis**:
    {analysis}

    **Owner Input**:
    {details}

    Create structured report with:
    1. Immediate care recommendations
    2. Food suggestions (base on life stage)
    3. Health monitoring checklist
    4. When to see a vet
    """
    return agent.run(prompt).content

# Initialize AI
pet_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions="Provide concise, actionable pet care advice. Prioritize common health issues."
)

# Image Upload
uploaded_image = st.file_uploader("Take/upload pet photo", type=["jpg", "jpeg", "png"])

# Essential Inputs
with st.form("pet_info"):
    age = st.number_input("Pet's age (months)", min_value=1, max_value=360, value=6)
    main_concern = st.selectbox(
        "Main concern",
        ("Healthy checkup", "Skin issues", "Digestion", "Behavior", "Weight")
    )
    current_food = st.text_input("Current food", placeholder="Brand/type")
    
    if st.form_submit_button("Generate Report"):
        if not uploaded_image:
            st.warning("Please upload a photo")
        else:
            with st.spinner("Creating report..."):
                try:
                    # Process image
                    img = Image.open(uploaded_image)
                    analysis = analyze_pet_image(img)
                    
                    # Prepare inputs
                    details = f"""
                    - Age: {age} months
                    - Concern: {main_concern}
                    - Current Food: {current_food}
                    """
                    
                    # Generate report
                    st.subheader("Basic Health Check")
                    st.write(analysis)
                    
                    report = generate_report(pet_agent, analysis, details)
                    st.subheader("Care Instructions")
                    st.markdown(report)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Always consult a veterinarian for serious health issues")
