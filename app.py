
import sys
import asyncio
import os

# Windows specific event loop policy not needed for sync invoke
# if sys.platform.startswith("win"):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import uuid
from core.research_agent import ResearchAgent
from core.document_generator import Generator
# nest_asyncio no longer needed

# Page Configuration
st.set_page_config(
    page_title="DeepResearch – Multi-Agent Research Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, minimal look
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Stricter Background Control for Cards/inputs to match config */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        color: #1a1a1a;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #000000;
        box-shadow: none;
    }

    /* Analyst Cards - Refined Shadow & Border */
    .analyst-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #f0f0f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    .analyst-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-color: #e0e0e0;
    }
    .analyst-name {
        font-weight: 600;
        font-size: 1.1rem;
        color: #111;
        margin-bottom: 0.25rem;
    }
    .analyst-role {
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .analyst-aff {
        font-size: 0.8rem;
        color: #888;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    .analyst-desc {
        font-size: 0.95rem;
        color: #333;
        line-height: 1.6;
    }
    
    /* Document Container */
    .report-container {
        padding: 3rem;
        background: white;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        border-radius: 4px;
        margin-top: 1rem;
    }
    
    /* Button overrides are handled by config.toml mostly, but for secondary buttons: */
    div[data-testid="column"] button {
        border: 1px solid #e0e0e0 !important;
        background: white !important;
        color: #333 !important;
    }
    div[data-testid="column"] button:hover {
        border-color: #000 !important;
        color: #000 !important;
        background: #fafafa !important;
    }

</style>
""", unsafe_allow_html=True)

# Helper function to ensure Document directory exists
def ensure_doc_dir():
    if not os.path.exists("Document"):
        os.makedirs("Document")

# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "analysts" not in st.session_state:
    st.session_state.analysts = None
if "research_active" not in st.session_state:
    st.session_state.research_active = False
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "agent_graph" not in st.session_state:
    st.session_state.agent_graph = None
    
# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    with st.container():
        topic = st.text_input("Research Topic", "The Future of AI Agents", 
                             help="Primary subject or theme you want the research team to investigate in depth.")
        max_analysts = st.slider("Number of Analysts", min_value=1, max_value=5, value=2,
                                help="Controls the size of your research team. More analysts provide broader perspectives but take longer.")
        template_prompt = st.text_area("Custom Instructions", height=150, 
                                      placeholder="Optional: Provide specific focus areas or guidelines for the research team.",
                                      help="Specific constraints, desired focus areas, or formatting requirements for the final report.")
    
    st.markdown("---")
    
    if st.button("Start Research", use_container_width=True):
        # Reset state for new research
        st.session_state.research_active = True
        st.session_state.final_report = None
        st.session_state.analysts = None
        st.session_state.agent_graph = None # Reset graph
        st.session_state.thread_id = str(uuid.uuid4()) # New thread
        st.rerun()


# --- Main Logic ---

st.title("DeepResearch – LangGraph Based Multi-Agent Research Engine")

# Function to run the agent steps synchronously
def run_research_step(initial=False, feedback=None):
    thread = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    try:
        if initial:
            # Import default template if custom one is not provided
            if not template_prompt:
                from core.prompts import template as default_template
                instructions = default_template
            else:
                instructions = template_prompt
            
            # Build graph
            agent_graph = ResearchAgent(instructions).build()
            st.session_state.agent_graph = agent_graph
            
            # Initial invoke
            result = agent_graph.invoke({
                "topic": topic,
                "max_analysts": max_analysts,
            }, thread)
            
            return result
        
        elif feedback is not None:
            agent_graph = st.session_state.agent_graph
            
            # Update State with feedback
            agent_graph.update_state(
                thread, 
                {"human_analyst_feedback": feedback}, 
                as_node="human_feedback"
            )
            
            result = agent_graph.invoke(None, thread)
            return result
            
        else:
            # Proceed without feedback (if empty)
            agent_graph = st.session_state.agent_graph
            agent_graph.update_state(
                 thread,
                 {"human_analyst_feedback": None}, # Explicitly None for "no feedback"
                 as_node="human_feedback"
            )
            result = agent_graph.invoke(None, thread)
            return result
            
    except Exception as e:
        st.error(f"Error during research execution: {e}")
        st.exception(e)
        return None



# No longer needed: run_async helper

# --- Application Flow ---

from dotenv import load_dotenv
load_dotenv()

if not st.session_state.research_active:
    st.markdown("""
    Welcome to DeepResearch. 
    
    This tool orchestrates a team of AI analysts to conduct deep research on any topic. 
    Configure your parameters in the sidebar and click **Start Research** to begin.
    """)

else:
    # Container for status updates
    status_container = st.container()

    if st.session_state.analysts is None:
        # Phase 1: Create Analysts
        with status_container:
            with st.status("Initializing Research Team...", expanded=True) as status:
                st.write("Analyzing research topic...")
                st.write("Selecting domain experts...")
                
                # Run the initial step
                result = run_research_step(initial=True)
                
                if result:
                    # Check if we are paused at human_feedback
                    analysts_data = result.get('analysts', [])
                    if analysts_data:
                        st.session_state.analysts = analysts_data
                        status.update(label="Team Assembled", state="complete", expanded=False)
                        st.rerun() # Rerun to show the analyst UI
                else:
                     status.update(label="Initialization Failed", state="error")
                     st.stop()

    elif st.session_state.final_report is None:
        # Phase 2: Analyst Review & Feedback
        
        st.subheader("Research Team")
        st.markdown("The following analysts have been selected to research your topic. You may provide guidance to refine their focus.")
        
        # Display Analysts in a grid
        cols = st.columns(len(st.session_state.analysts))
        
        for i, analyst in enumerate(st.session_state.analysts):
            with cols[i]:
                st.markdown(f"""
                <div class="analyst-card">
                    <div class="analyst-name" style="color: #2E74B5; font-size: 1.1rem; border-bottom: 2px solid #f0f0f0; padding-bottom: 0.5rem; margin-bottom: 1rem;">{analyst.role}</div>
                    <div class="analyst-desc">{analyst.description}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feedback Form
        with st.form("feedback_form"):
            st.subheader("Analyst Guidance")
            feedback = st.text_area("Guidance", label_visibility="collapsed",
                                   placeholder="Enter feedback or specific questions for the analysts...",
                                   help="Leave empty to proceed with the current research plan.")
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submitted = st.form_submit_button("Proceed")
            
            if submitted:
                feedback_val = feedback if feedback.strip() else None
                
                with st.status(" conducting research...", expanded=True) as status:
                    st.write(" conducting interviews with experts...")
                    st.write(" gathering external resources...")
                    st.write(" synthesizing findings...")
                    st.write(" compiling final report...")
                    
                    # Run the final step
                    result = run_research_step(feedback=feedback_val)
                    
                    if result:
                        st.session_state.final_report = result.get("final_report", "No report generated.")
                        status.update(label="Research Complete", state="complete", expanded=False)
                        st.rerun()
                    else:
                        status.update(label="Research Failed", state="error")


if st.session_state.final_report:
    # Phase 3: Final Report
    st.success("Research Report Generated Successfully")
    
    # Display Report
    with st.container():
        st.markdown(f'<div class="report-container">{st.session_state.final_report}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Downloads")
    
    # Ensure doc dir exists
    ensure_doc_dir()
    gen = Generator()
    
    col1, col2, col3 = st.columns(3)
    
    try:
        # PDF
        with col1:
            if st.button("Download PDF"):
                with st.spinner("Generating..."):
                    pdf_path = gen.generate_pdf(st.session_state.final_report)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Click to Download PDF",
                            data=f,
                            file_name="DeepResearch_Report.pdf",
                            mime="application/pdf"
                        )

        # DOCX
        with col2:
             if st.button("Download DOCX"):
                with st.spinner("Generating..."):
                    docx_path = gen.generate_doc(st.session_state.final_report)
                    with open(docx_path, "rb") as f:
                        st.download_button(
                            label="Click to Download DOCX",
                            data=f,
                            file_name="DeepResearch_Report.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        
        # PPTX
        with col3:
             if st.button("Download PPTX"):
                with st.spinner("Generating..."):
                    pptx_path = gen.generate_pptx(st.session_state.final_report)
                    with open(pptx_path, "rb") as f:
                        st.download_button(
                            label="Click to Download PPTX",
                            data=f,
                            file_name="DeepResearch_Report.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                        )
    except Exception as e:
        st.error(f"Error generating documents: {e}")
