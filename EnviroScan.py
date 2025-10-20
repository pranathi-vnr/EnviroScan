import streamlit as st
import pandas as pd
import os
import subprocess
import sys
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EnviroScan AI - Home",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    /* Main App Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom Headers */
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .header h1 {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header p {
        font-size: 1.25rem;
        color: #4f4f4f;
    }

    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1e1e1e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #0072ff;
        padding-bottom: 0.5rem;
    }

    /* Custom Cards for Features */
    .feature-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #0072ff;
    }
    .feature-card h4 {
        margin-top: 0;
        color: #1e1e1e;
        font-size: 1.25rem;
        font-weight: 600;
    }
    .feature-card p {
        color: #4f4f4f;
        flex-grow: 1;
    }
    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        margin-top: 1rem;
    }
    .status-ready {
        background-color: #e6f7f0;
        color: #00874e;
    }
    .status-pending {
        background-color: #fff4e5;
        color: #ff9800;
    }
    .status-available {
        background-color: #e9f5ff;
        color: #0072ff;
    }

    /* Pipeline Status Card */
    .pipeline-card {
        background-color: #fffbe6;
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Custom Button */
    .stButton>button {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
    }
    
    /* Success and Error Messages */
    .success-box {
        background-color: #e6f7f0;
        border-left: 5px solid #00874e;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #ffe6e6;
        border-left: 5px solid #d32f2f;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

</style>
""", unsafe_allow_html=True)

# --- PATHS AND CONFIGURATION ---
def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent

def check_pipeline_status():
    """Check if pipeline has been run and models are available"""
    status = {
        'models_ready': False,
        'output_files': [],
        'missing_files': [],
        'outputs_dir_exists': False
    }
    
    # Check if outputs directory exists
    outputs_dir = get_project_root() / "outputs"
    status['outputs_dir_exists'] = outputs_dir.exists()
    
    expected_files = [
        "outputs/pollution_source_model.joblib",
        "outputs/label_encoder.joblib",
        "outputs/scaler.joblib",
        "outputs/processed_data_for_eda.csv"
    ]
    
    for file in expected_files:
        file_path = get_project_root() / file
        if file_path.exists():
            status['output_files'].append(file)
        else:
            status['missing_files'].append(file)
    
    status['models_ready'] = len(status['missing_files']) == 0
    return status

def run_pipeline():
    """Run the data pipeline with enhanced error handling"""
    try:
        # Check if run_pipeline.py exists
        pipeline_script = get_project_root() / "run_pipeline.py"
        if not pipeline_script.exists():
            return False, f"Pipeline script not found at: {pipeline_script}"
        
        with st.spinner('🚀 **Running Data & Modeling Pipeline...** This may take a few minutes. Please wait.'):
            # Run the pipeline
            result = subprocess.run(
                [sys.executable, str(pipeline_script)],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=get_project_root()  # Run from project root
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                error_msg = f"Exit code: {result.returncode}\nError: {result.stderr}"
                return False, error_msg
                
    except subprocess.TimeoutExpired:
        return False, "Pipeline timed out after 10 minutes. The process took too long."
    except FileNotFoundError:
        return False, "Python interpreter or pipeline script not found."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def check_api_key():
    """Check if OpenWeather API key is configured"""
    env_file = get_project_root() / ".env"
    if env_file.exists():
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                if "OPENWEATHER_API_KEY" in content and "your_key_here" not in content:
                    return True
        except:
            pass
    return False

def load_locations_data():
    """Load and validate locations data"""
    try:
        locations_path = get_project_root() / "data" / "locations.csv"
        if not locations_path.exists():
            return None, "Locations file not found"
        
        df_locations = pd.read_csv(locations_path)
        
        # Validate required columns
        required_columns = ['name', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df_locations.columns]
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Validate data
        if df_locations.empty:
            return None, "Locations file is empty"
        
        # Check for valid coordinates
        if (df_locations['latitude'].abs() > 90).any() or (df_locations['longitude'].abs() > 180).any():
            return None, "Invalid coordinates found in locations file"
            
        return df_locations, None
        
    except Exception as e:
        return None, f"Error loading locations: {str(e)}"

# --- HEADER SECTION ---
st.markdown("""
<div class="header">
    <h1>Welcome to EnviroScan AI 💨</h1>
    <p>AI-Powered Pollution Source Identification using Geospatial Analytics</p>
</div>
""", unsafe_allow_html=True)

# --- PIPELINE STATUS & SETUP ---
st.markdown('<h2 class="section-header">🚀 Quick Setup & Status</h2>', unsafe_allow_html=True)

# Check API key status
api_key_configured = check_api_key()

# Check pipeline status
pipeline_status = check_pipeline_status()

if not pipeline_status['models_ready']:
    # Show setup requirements
    st.markdown("""
    <div class="pipeline-card">
        <h3 style="margin-top:0;">⚠️ Setup Required!</h3>
        <p>Before using the prediction features, complete the setup steps below:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup checklist
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Setup Checklist")
        
        # API Key check
        if api_key_configured:
            st.success("✅ OpenWeather API Key Configured")
        else:
            st.error("❌ OpenWeather API Key Missing")
            
        # Outputs directory check
        if pipeline_status['outputs_dir_exists']:
            st.success("✅ Outputs Directory Exists")
        else:
            st.warning("⚠️ Outputs Directory Not Found")
    
    with col2:
        st.subheader("🔧 Required Files")
        if pipeline_status['output_files']:
            st.success(f"✅ {len(pipeline_status['output_files'])} files ready")
        if pipeline_status['missing_files']:
            st.error(f"❌ {len(pipeline_status['missing_files'])} files missing")
    
    # Pipeline run section
    st.markdown("---")
    st.subheader("🛠️ Run Data Pipeline")
    
    if not api_key_configured:
        st.warning("""
        **API Key Required:** Please configure your OpenWeather API key before running the pipeline.
        
        1. Create a `.env` file in the project root
        2. Add: `OPENWEATHER_API_KEY="your_actual_key_here"`
        3. Remove any placeholder values
        """)
    
    col_run1, col_run2 = st.columns([3, 1])
    
    with col_run1:
        st.info("""
        **What the pipeline does:**
        - Collects environmental data from OpenWeather API
        - Processes and engineers features
        - Trains machine learning models
        - Generates analytics and visualizations
        """)
    
    with col_run2:
        if st.button("▶️ Run Data Pipeline", 
                    type="primary" if api_key_configured else "secondary",
                    disabled=not api_key_configured,
                    use_container_width=True,
                    help="Run the complete data processing and model training pipeline"):
            
            success, output = run_pipeline()
            
            if success:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ Pipeline Successful!</h4>
                    <p>All models trained and ready. Refreshing...</p>
                </div>
                """, unsafe_allow_html=True)
                st.toast("Pipeline complete! Refreshing...")
                st.rerun()
            else:
                st.markdown("""
                <div class="error-box">
                    <h4>❌ Pipeline Failed</h4>
                    <p>See error details below:</p>
                </div>
                """, unsafe_allow_html=True)
                st.code(output, language='bash')
    
    # Detailed file status
    with st.expander("📁 Detailed File Status", expanded=False):
        col_status1, col_status2 = st.columns(2)
        
        with col_status1:
            st.write("**✅ Available Files:**")
            if pipeline_status['output_files']:
                for file in pipeline_status['output_files']:
                    st.markdown(f"`{file}`")
            else:
                st.write("No files available")
        
        with col_status2:
            st.write("**❌ Missing Files:**")
            if pipeline_status['missing_files']:
                for file in pipeline_status['missing_files']:
                    st.markdown(f"`{file}`")
            else:
                st.write("All files available")
                
else:
    # System is ready
   st.markdown("""
    <div style="
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    ">
        <h3 style="color: #1976d2; margin-top: 0;">✅ System Ready!</h3>
        <p style="color: #1565c0; margin-bottom: 0;">All models are trained and available. You can now access all platform features from the sidebar.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- INTRODUCTION ---
st.markdown('<h2 class="section-header">💡 About the Platform</h2>', unsafe_allow_html=True)
col1, col2 = st.columns([1.8, 1], gap="large")

with col1:
    st.markdown("""
    ### Revolutionizing Environmental Monitoring
    
    EnviroScan AI leverages sophisticated machine learning models to analyze complex environmental data. 
    By identifying pollution sources in near real-time, we empower authorities, researchers, and urban 
    planners to make faster, more effective decisions for a cleaner, healthier planet.
    
    **Key Capabilities:**
    - 🎯 **Source Identification**: Accurately pinpoint pollution sources using AI
    - 📊 **Real-time Analytics**: Monitor and analyze environmental trends
    - 🗺️ **Geospatial Visualization**: Interactive maps showing pollution hotspots
    - 🔮 **Predictive Modeling**: Forecast pollution scenarios and impacts
    
    Our platform provides an intuitive interface to interact with powerful analytics, turning raw data 
    into actionable intelligence for environmental protection and urban planning.
    """)
    
    if pipeline_status['models_ready']:
        st.success("**👈 Select a page from the sidebar to begin your analysis!**", icon="🧭")
    else:
        st.warning("**⬆️ Complete the setup steps above to unlock all features.**", icon="⚙️")

with col2:
    # You can add an image here when deploying
    st.info("""
    **🌍 Environmental Impact**
    
    - **Urban Planning**: Optimize city layouts for better air quality
    - **Policy Making**: Data-driven environmental regulations
    - **Public Health**: Identify and mitigate pollution hotspots
    - **Research**: Advanced analytics for environmental studies
    """)

st.markdown("---")

# --- CORE FEATURES ---
st.markdown('<h2 class="section-header">✨ Core Features</h2>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="large")

features = [
    {
        "col": c1, 
        "title": "🌍 Pollution Dashboard",
        "desc": "Visualize geospatial pollution hotspots, analyze trends over time, and filter data by city or predicted source. Interactive maps and comprehensive analytics.",
        "status": "ready" if pipeline_status['models_ready'] else "pending",
        "icon": "📊"
    },
    {
        "col": c2, 
        "title": "🔬 Live Prediction", 
        "desc": "Interact directly with our AI model. Input custom environmental parameters to simulate scenarios and receive instant pollution source predictions with confidence scores.",
        "status": "ready" if pipeline_status['models_ready'] else "pending",
        "icon": "🤖"
    },
    {
        "col": c3, 
        "title": "ℹ️ About & Methodology",
        "desc": "Discover the scientific methodology, data sources, machine learning models, and technologies powering the EnviroScan AI platform.",
        "status": "available",
        "icon": "📚"
    }
]

for feature in features:
    with feature["col"]:
        status_text = {
            "ready": "✅ Ready to Use",
            "pending": "⚠️ Run Pipeline First", 
            "available": "📖 Always Available"
        }.get(feature["status"])
        
        st.markdown(f"""
        <div class="feature-card">
            <h4>{feature['icon']} {feature['title']}</h4>
            <p>{feature['desc']}</p>
            <div class="status-badge status-{feature['status']}">
                {status_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- DATA SCOPE & INSTRUCTIONS ---
st.markdown('<h2 class="section-header">📋 Setup & Data Scope</h2>', unsafe_allow_html=True)
sc1, sc2 = st.columns(2, gap="large")

with sc1:
    st.subheader("🔧 Setup Instructions")
    
    instructions = """
    **1. API Configuration**
    - Sign up at [OpenWeatherMap](https://openweathermap.org/api)
    - Get your free API key
    - Create `.env` file with: `OPENWEATHER_API_KEY="your_key"`
    
    **2. Run Pipeline** 
    - Click "Run Data Pipeline" button above
    - Or run manually: `python run_pipeline.py`
    - Wait for completion (2-5 minutes)
    
    **3. Explore Features**
    - Use sidebar navigation
    - Start with Pollution Dashboard
    - Try Live Prediction scenarios
    """
    
    st.info(instructions, icon="⚙️")

with sc2:
    st.subheader("🌐 Monitoring Scope")
    
    df_locations, locations_error = load_locations_data()
    
    if df_locations is not None:
        num_cities = df_locations['name'].nunique()
        
        st.metric(
            label="Cities Currently Monitored", 
            value=f"{num_cities} Cities",
            help="Number of cities in our monitoring network"
        )
        
        st.success(f"""
        **Global Coverage:** Our network monitors **{num_cities} cities** worldwide, 
        providing comprehensive environmental data for analysis and prediction.
        """, icon="🌍")
        
        with st.expander("📋 View Monitored Cities"):
            st.dataframe(
                df_locations,
                use_container_width=True,
                hide_index=True
            )
            
    else:
        st.error(f"""
        **Configuration Needed:** {locations_error}
        
        Please ensure the locations file exists at `data/locations.csv` with the required format.
        """, icon="⚠️")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem 0;'>
    <p><strong>EnviroScan AI v2.1 | Environmental Intelligence Platform</strong></p>
    <p style='font-size: 0.9em;'>Advanced machine learning for pollution source identification and environmental monitoring</p>
    <p style='font-size: 0.8em;'>For research, policy-making, and environmental protection</p>
</div>
""", unsafe_allow_html=True)