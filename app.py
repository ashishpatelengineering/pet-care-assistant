import streamlit as st
from PIL import Image
from phi.agent import Agent
from phi.model.google import Gemini
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API key from Streamlit Secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Configure API Key
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# Page Configuration
st.set_page_config(
    page_title="Pet Wellness Assistant",
    page_icon="🐾",
    layout="centered"
)

st.title("Pet Wellness Assistant")
st.header("Helping you understand your pet's well-being 🐶🐱")

def analyze_pet_image(api_key, prompt, image):
    """Analyze pet image using Gemini AI."""
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
    response = model.generate_content([prompt, image])
    return response.text

def initialize_agent():
    return Agent(
        name="Pet Wellness Advisor",
        model=Gemini(id="gemini-2.0-flash-exp"),  
        instructions=[
            "You are a pet wellness advisor providing insights on pet health, breed identification, and care.",
            "Assess visible health indicators such as skin condition, weight, and coat quality.",
            "Give general advice on pet nutrition, exercise, and wellness based on the image.",
            "Always recommend professional veterinary care if necessary.",
            "Format responses clearly and user-friendly for pet owners.",
        ],
        markdown=True
    )

# Initialize the Agent
pet_agent = initialize_agent()

# File Uploader
image_file = st.file_uploader("Upload a clear photo of your pet for wellness analysis", type=["jpg", "jpeg", "png"], help="Ensure good lighting and clarity for best analysis")

# User Inputs
pet_age = st.number_input("Pet's Age (in months)", min_value=1, max_value=360, value=6)
primary_concern = st.selectbox("Primary Concern", ["Routine checkup", "Skin & Coat", "Digestion", "Behavior", "Weight Management"])
current_diet = st.text_input("Current Diet", placeholder="Enter brand/type of food")
owner_query = st.text_area(
    "Any specific questions or concerns?",                 
    placeholder="Ask any wellness-related questions about your pet",
    help="The AI will provide customized insights based on your pet's image and details.")

if image_file:
    try:
        # Open and display the uploaded image
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Pet Image", use_column_width=True)
        
        # Get AI response using the image
        with st.spinner("Analyzing your pet's wellness..."):
            prompt = "Analyze this pet photo for wellness assessment. Consider breed, weight, coat condition, and health indicators."
            analysis = analyze_pet_image(API_KEY, prompt, image)
            
            prompt_details = f"""
            Pet Age: {pet_age} months
            Concern: {primary_concern}
            Diet: {current_diet}
            {owner_query}
            Provide general wellness insights and care recommendations.
            """
            
            response = pet_agent.run(prompt_details, image=image)
        
        st.subheader("Pet Wellness Overview")
        st.write(analysis)
        st.subheader("Personalized Wellness Advice")
        st.markdown(response.content)
        
    except Exception as e:
        st.error(f"Error: Unable to process image. {e}")

# Footer
st.markdown("---")
st.caption("Note: This tool provides general wellness insights. Always consult a veterinarian for professional medical advice.")
