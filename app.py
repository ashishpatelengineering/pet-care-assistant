import streamlit as st
from PIL import Image
import google.generativeai as genai
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.firecrawl import FirecrawlTools

# Configure API Keys
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Page Setup
st.set_page_config(page_title="Pet Care Assistant", page_icon="🐶", layout="centered")
st.title("Pet Health Check")

def analyze_pet_image(image):
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content([
        "Analyze this pet photo for:",
        "- Breed (confidence %)",
        "- Estimated age range",
        "- Weight assessment", 
        "- Visible health indicators",
        "Format: Clear bullet points",
        image
    ])
    return response.text

def generate_care_plan(agent, analysis, details):
    prompt = f"""
    **Pet Analysis**:
    {analysis}

    **Owner Input**:
    {details}

    Create care plan with:
    1. Nutrition (specific food types)
    2. Exercise recommendations
    3. Health monitoring checklist
    4. Immediate action items
    """
    return agent.run(prompt).content

# Initialize AI Agent
pet_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[FirecrawlTools(api_key=st.secrets["FIRCRAWL_API_KEY"])],
    instructions="Provide practical, veterinary-verified advice in clear sections."
)

# Image Upload
image_file = st.file_uploader("Upload pet photo", type=["jpg", "jpeg", "png"])
if image_file:
    st.image(Image.open(image_file), width=300)

# Simple Form
with st.form("pet_details"):
    st.subheader("Pet Information")
    age_months = st.number_input("Age (months)", min_value=1, max_value=360, value=12)
    health_notes = st.text_input("Main health concern", placeholder="Skin, digestion, etc.")
    diet_notes = st.text_area("Food preferences/allergies", height=80)
    
    if st.form_submit_button("Get Care Plan"):
        if not image_file:
            st.warning("Please upload a photo first")
        else:
            with st.spinner("Analyzing..."):
                try:
                    # Image Analysis
                    img = Image.open(image_file)
                    analysis = analyze_pet_image(img)
                    
                    # Prepare Details
                    details = f"""
                    - Age: {age_months} months
                    - Health Focus: {health_notes}
                    - Diet Notes: {diet_notes}
                    """
                    
                    # Generate Recommendations
                    st.subheader("Health Assessment")
                    st.markdown(f"**Basic Analysis**\n{analysis}")
                    
                    care_plan = generate_care_plan(pet_agent, analysis, details)
                    st.subheader("Care Plan")
                    st.markdown(care_plan)
                    
                except Exception as e:
                    st.error(f"Error generating plan: {str(e)}")

# Footer
st.markdown("---")
st.caption("Professional veterinary advice should always be sought for serious health concerns")
