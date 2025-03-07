import streamlit as st
from PIL import Image
import google.generativeai as genai
from phi.agent import Agent
from phi.model.google import Gemini

# Configure API Keys
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Page Setup
st.set_page_config(page_title="PawCheck Pro", page_icon="🐾", layout="centered")
st.title("PawCheck Pro")
st.caption("AI-Powered Pet Health Scanning")

def analyze_pet_health(image):
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content([
        Image.open(image),
        "Analyze this pet photo for key health indicators:",
        "1. Body Condition Score (1-9 scale)",
        "2. Coat/skin health (poor/fair/good)",
        "3. Visible signs of discomfort/pain",
        "Format: Concise bullet points"
    ])
    return response.text

def generate_health_report(agent, analysis, details):
    prompt = f"""
    **Animal Profile**:
    - Species: {details['species']}
    - Age: {details['age']} months
    - Owner's Concern: {details['concern']}

    **Visual Analysis**:
    {analysis}

    Create professional report:
    1. Health Summary (3 key points)
    2. Immediate Care Recommendations
    3. Warning Signs to Watch
    4. Recommended Follow-up Actions
    """
    return agent.run(prompt).content

# Initialize AI
med_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions="Provide veterinary-style advice in clear, actionable terms."
)

# --- Main Interface ---
st.subheader("Pet Information")
species = st.radio("Species", ("🐕 Dog", "🐈 Cat"), horizontal=True)
pet_name = st.text_input("Name (optional)", help="For personalized report")
age = st.slider("Age in months", 1, 360, 24)
concern = st.selectbox(
    "Main Health Concern",
    ("General Wellness Check", "Skin/Coat Issues", "Low Energy Levels", 
     "Eating Difficulties", "Behavior Changes", "Mobility Concerns")
)

st.divider()

st.subheader("Health Scan Upload")
uploaded_image = st.file_uploader(
    "Upload clear pet photo (JPEG/PNG)",
    type=["jpg", "jpeg", "png"],
    help="Full body profile with good lighting"
)

# --- Report Generation ---
if uploaded_image and st.button("Generate Health Report"):
    with st.spinner("Analyzing..."):
        try:
            # Display preview
            st.image(uploaded_image, width=300, caption="Uploaded Pet Photo")
            
            # Process analysis
            analysis = analyze_pet_health(uploaded_image)
            details = {
                "species": "Dog" if "Dog" in species else "Cat",
                "age": age,
                "concern": concern
            }
            
            # Generate report
            report = generate_health_report(med_agent, analysis, details)
            
            # Display results
            st.divider()
            report_title = f"{pet_name}'s Health Report" if pet_name else "Pet Health Report"
            st.subheader(report_title)
            st.markdown(report)
            
            # Professional CTA
            st.divider()
            st.success("**Need Professional Review?**")
            st.markdown("""
            Get verified veterinary consultation within 1 hour:
            - Video call with certified vet ($29)
            - Priority message review ($15)
            """)
            st.button("Connect with Veterinarian →")
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

# Footer
st.divider()
st.markdown("""
**Trusted By**  
🐾 PawsCare Clinics  |  🏥 UrbanVet Network  |  🦴 SafePet Insurance  
*Enterprise solutions available for clinics and shelters*
""")
