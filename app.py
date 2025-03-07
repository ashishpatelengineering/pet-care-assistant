import streamlit as st
from PIL import Image
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.firecrawl import FirecrawlTools
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API Keys
FIRCRAWL_API_KEY = st.secrets["FIRCRAWL_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Page Configuration
st.set_page_config(
    page_title="Pet Care Expert",
    layout="centered",
    page_icon="🐾"
)

st.title("Smart Pet Care Assistant")

def get_gemini_response(image):
    """Enhanced image analysis prompt"""
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
    prompt = """Analyze this pet photo and provide:
    1. Breed identification with confidence level
    2. Estimated age range
    3. Visible weight assessment
    4. Coat/skin condition observations
    5. Notable physical features
    6. Immediate care recommendations"""
    
    response = model.generate_content([prompt, image])
    return response.text

def initialize_agent():
    return Agent(
        name="Pet Care Expert",
        model=Gemini(id="gemini-2.0-flash-exp"),
        instructions=[
            "You're a veterinary-trained AI that provides evidence-based pet care advice.",
            "Prioritize observable factors from images and owner-reported information.",
            "Recommend products only from certified pet care brands and veterinary-approved sources.",
            "Always consider breed-specific needs and potential health risk factors.",
            "Present information in clear sections with emoji icons for readability.",
        ],
        tools=[FirecrawlTools(api_key=FIRCRAWL_API_KEY)],
        markdown=True
    )

pet_care_agent = initialize_agent()

# Image Analysis Section
with st.expander("📸 Pet Photo Analysis", expanded=True):
    image_file = st.file_uploader(
        "Upload clear pet photo (JPEG/PNG)",
        type=["jpg", "jpeg", "png"],
        help="Clear face/body shots work best for accurate analysis"
    )
    
    if image_file:
        try:
            image = Image.open(image_file)
            st.image(image, use_column_width=True)
            
            with st.spinner("🔍 Analyzing pet characteristics..."):
                analysis = get_gemini_response(image)
                st.subheader("Initial Health Assessment")
                st.write(analysis)
                
        except Exception as e:
            st.error(f"Image processing error: {str(e)}")

# Care Recommendations Section
with st.expander("💬 Personalized Care Consultation", expanded=True):
    pet_profile = st.text_input("Pet's name (optional)", help="For personalized recommendations")
    
    col1, col2 = st.columns(2)
    with col1:
        pet_age = st.number_input("Approximate age", min_value=0, max_value=30, help="Years")
        diet_needs = st.multiselect(
            "Dietary considerations",
            ["Weight Management", "Allergies", "Puppy/Kitten", "Senior", "Prescription Diet"]
        )
        
    with col2:
        health_priority = st.selectbox(
            "Primary health focus",
            ("Preventive Care", "Skin/Coat", "Dental", "Mobility", "Behavioral")
        )
        observed_behavior = st.text_area(
            "Recent changes or concerns",
            placeholder="e.g., itching, lethargy, appetite changes",
            height=100
        )

    if st.button("🚀 Generate Care Plan", type="primary"):
        if not image_file:
            st.warning("Please upload a pet photo first")
        else:
            try:
                consultation_prompt = f"""
                **Pet Profile**
                - Name: {pet_profile}
                - Age: {pet_age} years
                - Dietary Needs: {', '.join(diet_needs)}
                - Health Priority: {health_priority}
                - Observations: {observed_behavior}
                
                **Owner Request**
                Comprehensive care plan covering:
                1. Nutrition recommendations
                2. Preventive care measures
                3. Behavioral/environmental suggestions
                4. Recommended health monitoring
                
                Include product links where appropriate.
                """
                
                with st.spinner("📋 Creating customized care plan..."):
                    response = pet_care_agent.run(consultation_prompt, image=image)
                    
                st.subheader("📝 Your Pet's Care Plan")
                st.markdown(response.content)
                
            except Exception as error:
                st.error(f"Analysis error: {str(error)}")

# Add subtle branding
st.markdown("---")
st.caption("🐾 Powered by Veterinary AI • Trusted Pet Care Since 2024")
