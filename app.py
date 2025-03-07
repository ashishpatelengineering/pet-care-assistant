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
    """Image analysis prompt"""
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
            "Combine image analysis with owner-provided information for recommendations.",
            "Recommend products only from certified sources.",
            "Present information in clear sections with emoji icons.",
        ],
        tools=[FirecrawlTools(api_key=FIRCRAWL_API_KEY)],
        markdown=True
    )

pet_care_agent = initialize_agent()

# Image Upload Section
with st.expander("📸 Upload Pet Photo", expanded=True):
    image_file = st.file_uploader(
        "Upload clear pet photo (JPEG/PNG)",
        type=["jpg", "jpeg", "png"],
        help="Clear face/body shots work best"
    )
    if image_file:
        image = Image.open(image_file)
        st.image(image, use_column_width=True)

# Consultation Form
with st.expander("💬 Pet Details & Consultation", expanded=True):
    pet_name = st.text_input("Pet's name (optional)")
    
    col1, col2 = st.columns(2)
    with col1:
        pet_age = st.number_input("Age", min_value=0, max_value=30)
        diet_needs = st.multiselect(
            "Dietary needs",
            ["Weight Management", "Allergies", "Puppy/Kitten", "Senior", "Prescription Diet"]
        )
        
    with col2:
        health_priority = st.selectbox(
            "Health focus",
            ("Preventive Care", "Skin/Coat", "Dental", "Mobility", "Behavioral")
        )
        observations = st.text_area(
            "Behavior/Changes",
            placeholder="e.g., itching, lethargy, appetite changes",
            height=100
        )

    if st.button("🚀 Generate Full Report", type="primary"):
        if not image_file:
            st.warning("Please upload a pet photo")
        else:
            try:
                with st.spinner("🔍 Analyzing..."):
                    # Get image analysis
                    image = Image.open(image_file)
                    analysis = get_gemini_response(image)
                    
                    # Create combined prompt
                    consultation_prompt = f"""
                    **Visual Analysis**:
                    {analysis}

                    **Owner-Reported Details**:
                    - Name: {pet_name}
                    - Age: {pet_age} years
                    - Dietary Needs: {', '.join(diet_needs)}
                    - Health Priority: {health_priority}
                    - Observations: {observations}

                    **Required**:
                    1. Combined health assessment
                    2. Nutrition plan
                    3. Care recommendations
                    4. Monitoring checklist
                    """
                    
                    # Get care plan
                    response = pet_care_agent.run(consultation_prompt)
                
                st.subheader("Initial Health Assessment")
                st.write(analysis)
                st.subheader("📝 Comprehensive Care Plan")
                st.markdown(response.content)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("🐾 Powered by Veterinary AI • Trusted Pet Care Since 2024")
