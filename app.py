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
st.caption("Smart Pet Health Scanning for Modern Owners")

def analyze_pet_health(image):
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content([
        Image.open(image),
        "Analyze this pet photo for key health indicators:",
        "1. Body Condition Score (1-9 scale)",
        "2. Coat/skin health (poor/fair/good)",
        "3. Eye/nose/ear abnormalities",
        "4. Visible signs of discomfort",
        "Format: Concise bullet points"
    ])
    return response.text

def generate_health_report(agent, analysis, details):
    prompt = f"""
    **Animal**: {details['species']} ({details['age']} months)
    **Owner Concerns**: {details['concern']}

    **Visual Analysis**:
    {analysis}

    Create veterinary-style report:
    1. Health Summary (3 key points)
    2. Home Care Protocol (daily/weekly)
    3. Red Flag Alerts (when to seek vet)
    4. Preventive Recommendations
    """
    return agent.run(prompt).content

# Initialize AI
med_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions="You are a veterinary assistant. Provide professional but owner-friendly advice."
)

# --- Core Interface ---
with st.container():
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("1. Pet Profile")
        species = st.radio("Species", ("Dog", "Cat"), horizontal=True)
        pet_name = st.text_input("Name", placeholder="Optional")
        age = st.slider("Age (months)", 1, 360, 24)
        concern = st.selectbox(
            "Main Concern",
            ("General Checkup", "Skin/Coat Issues", "Low Energy", 
             "Eating Problems", "Behavior Changes")
        )
        
    with col2:
        st.subheader("2. Health Scan")
        uploaded_image = st.camera_input("Take clear photo", 
            help="Full body profile, good lighting")

# --- Report Generation ---
if uploaded_image and st.button("Generate Health Report"):
    with st.spinner("Analyzing..."):
        try:
            # Process inputs
            analysis = analyze_pet_health(uploaded_image)
            details = {
                "species": species,
                "age": age,
                "concern": concern
            }
            
            # Generate report
            report = generate_health_report(med_agent, analysis, details)
            
            # Display results
            st.divider()
            if pet_name:
                st.subheader(f"{pet_name}'s Health Report")
            else:
                st.subheader("Pet Health Report")
                
            st.image(uploaded_image, width=300)
            st.markdown(report)
            
            # Business CTA
            st.divider()
            st.success("**Professional Follow-Up Available**")
            st.write("Book a video consultation with certified veterinarians:")
            st.page_link("https://example.com/pro_consults", label="Schedule Now →")
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# Footer
st.divider()
st.markdown("""
**Enterprise Solutions Available**  
*For clinics, shelters, and pet care brands*  
→ Custom white-label versions  
→ API integration  
→ Bulk health screening  
""")
