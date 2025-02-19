import streamlit as st
import requests

#source ~/tr_inf/bin/activate

# Set page title and layout
st.set_page_config(
    page_title="Transplant Infections AI Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Chat UI
st.markdown("""
    <style>
    /* General Background */
    body {
        background-color: #f8f9fa;  /* Off-white background */
        color: #222222;  /* Deep black text */
    }
    
    .stApp {
        background-color: #ffffff;  /* White app container */
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }

    /* Input Box */
    .stTextInput > div > div > input {
        border: 2px solid #57068c;  /* NYU Purple border */
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        color: #222222;  /* Deep black text */
        background-color: #ffffff;  /* Keep input white */
    }

    /* Buttons */
    .stButton > button {
        background-color: #57068c;  /* NYU Purple */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background 0.3s, color 0.3s;
    }

    .stButton > button:hover {
        background-color: #3e0568;  /* Darker NYU Purple */
        color: #ffffff;
    }

    /* Chat Message Box */
    .message-box {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 16px;
    }

    /* User Message Styling */
    .user-message {
        background-color: #f0e6f9;  /* Soft lavender */
        color: #222222;  /* Black text */
        text-align: right;
        padding: 10px;
        border-radius: 8px;
    }

    /* Bot (AI) Message Styling */
    .bot-message {
        background-color: #e0e0e0;  /* Light gray */
        color: #222222;
        text-align: left;
        padding: 10px;
        border-radius: 8px;
    }

    /* Improve Visibility of Markdown Text */
    .stMarkdown {
        color: #222222 !important;  /* Ensures deep black text */
    }

    /* Ensure Captions Are Visible */
    .stCaption {
        color: #555555 !important;  /* Dark gray */
    }

    /* Tabs Styling */
    div[data-baseweb="tab-list"] button {
        font-size: 18px !important;
        font-weight: bold !important;
        color: #222222 !important;
        background-color: #ffffff !important;
        border-bottom: 3px solid #57068c !important;
        padding: 12px 20px;
    }

    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #f0e6f9 !important;  /* Soft lavender for selected tab */
        color: #222222 !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown("<h1 style='text-align: center; color: #008CBA;'>💬 Transplant Infections AI Chat</h1>", unsafe_allow_html=True)
st.write("---")

# API URL - for railway deployment
API_URL = "web-production-7c3ab.up.railway.app"

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Default query text
default_query = "Which viruses are important in xenotransplantation?"

# Initialize session state for query input
if "query" not in st.session_state:
    st.session_state.query = default_query

# Function to clear default text when button is clicked
def clear_query():
    st.session_state.query = ""  # Clear text box

# 📌 **TAB INTERFACE**
# Inject custom CSS to increase tab font size
st.markdown("""
    <style>
        div[data-baseweb="tab-list"] button {
            font-size: 25px !important;  /* Adjust font size */
            font-weight: bold !important;
            padding: 10px 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Create tabs with larger font
tab1, tab2 = st.tabs(["💬 Chat", "📄 Publications"])

# 📌 **TAB 1: CHAT INTERFACE**
with tab1:
    st.markdown("<h3>🗨️ Chat History</h3>", unsafe_allow_html=True)
    chat_container = st.container()

    # Display previous messages
    for entry in st.session_state.chat_history:
        role, text = entry["role"], entry["text"]
        if role == "user":
            st.markdown(f"<div class='message-box user-message'><b>You:</b> {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='message-box bot-message'><b>AI:</b> {text}</div>", unsafe_allow_html=True)

    # User Input Section
    st.markdown("<h3>🔍 Ask a Question</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])  # Two-column layout (Input & Clear Button)

    with col2:
        st.button("🗑️ Clear", on_click=clear_query)  # Clear button outside the form

    # ✅ Use a form to allow Enter key submission
    with col1, st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input(
            "Query",
            value=st.session_state.query,
            key="query_input",
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button("📤")  # Send button with arrow icon

    # Process User Query
    if submit_button and query.strip():
        # Store user question in chat history
        st.session_state.chat_history.append({"role": "user", "text": query})

        # Send request to FastAPI with chat history
        with st.spinner("🔎 Searching for the best answer... Please wait."):
            try:
                # Send full chat history to API
                payload = {"query": query, "history": st.session_state.chat_history}
                response = requests.post(API_URL, json=payload, timeout=15)

                if response.status_code == 200:
                    answer = response.json().get("answer", "No response received.")
                    st.session_state.chat_history.append({"role": "bot", "text": answer})

                    # Refresh UI
                    st.rerun()
                else:
                    st.error(f"❌ API Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("⚠️ FastAPI server is not running. Please start it first.")
            except requests.exceptions.Timeout:
                st.error("⏳ The request took too long. Please try again later.")
            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ Request failed: {e}")

# 📌 **TAB 2: PUBLICATIONS LIST**
with tab2:
    st.markdown("<h3>📄 Publications Used in Model</h3>", unsafe_allow_html=True)

    # Example list of publications
    publications = [
        "Multidrug-resistant bacteria in solid organ transplant recipients",
"Multidrug-Resistant Gram-Negative Bacteria Infections in Solid Organ Transplantation",
"Infection in Organ Transplantation – rather comprehensive, covers bacteria/fungi/viruses",
"Outcome of Transplantation Using Organs From Donors Infected or Colonized With Carbapenem-Resistant Gram-Negative Bacteria",
"Infections in Solid-Organ Transplant Recipients",
"Fungal infections in solid organ transplantation: An update on diagnosis and treatment",
"Invasive fungal infections in solid organ transplant recipients",
"Diagnostic and therapeutic approach to infectious diseases in solid organ transplant recipients",
"Nocardia Infections in Solid Organ Transplantation",
"Epidemiology and Clinical Manifestations of Listeria monocytogenes Infection",
"Mycobacterium tuberculosis after solid organ transplantation: A review of more than 2000 cases",
"Pathophysiology and Immunology (of Tuberculosis Infection)",
"Tuberculosis and Organ Transplantation",
"Aspergillus-related pulmonary diseases in lung transplantation",
"Invasive Aspergillosis after Renal Transplantation",
"Invasive aspergillosis in liver transplant recipients",
"Mucormycosis",
"Mucormycosis in renal transplant recipients: review of 174 reported cases",
"Mucormycosis in lung transplant recipients: A systematic review of the literature and a case series",
"Basic Principles of the virulence of Cryptococcus",
"Virulence mechanisms and Cryptococcus neoformans pathogenesis",
"Pneumocystis jiroveci",
"The Pathogenesis and Diagnosis of Pneumocystis jiroveci Pneumonia",
"Respiratory Viral Infections in Solid Organ and Hematopoietic Stem Cell Transplantation",
"Community-Acquired Respiratory Viruses in Transplant Patients: Diversity, Impact, Unmet Clinical Needs",
"Influenza and other respiratory virus infections in solid organ transplant recipients",
"Respiratory Syncytial Virus: A Comprehensive Review of Transmission, Pathophysiology, and Manifestation",
"Cytomegalovirus infection in solid organ transplant recipients",
"Common viral infections in kidney transplant recipients",
"Immunobiology and pathogenesis of hepatitis B virus infection",
"Hepatitis C virus infection",
"Solid Organ Transplant and Parasitic Diseases: A Review of the Clinical Cases in the Last Two Decades",
"Helminths in Organ Transplantation",
"Bloodstream infections after solid-organ transplantation",
"Cryptosporidium infection in solid organ transplantation",
"American trypanosomiasis (Chagas disease) in solid organ transplantation",
"Infectious disease risks in xenotransplantation",
"Infectious Diseases and Clinical Xenotransplantation",
"Risks of Infectious Disease in Xenotransplantation",
"Infection and clinical xenotransplantation: Guidance from the Infectious Disease Community of Practice of the American Society of Transplantation",
"Moving xenotransplantation from bench to bedside: Managing infectious risk",
"Xenotransplantation — A special case of One Health",
"Porcine Deltacoronaviruses: Origin, Evolution, Cross-Species Transmission and Zoonotic Potential",
"Potential zoonotic swine enteric viruses: The risk ignored for public health",
"KDIGO clinical practice guideline for the care of kidney transplant recipients",
"Guidance on the Use of Increased Infectious Risk Donors for Organ Transplantation",
"OPTN Policy",
"Foreword: 4th edition of the American Society of Transplantation Infectious Diseases Guidelines",
"Solid organ transplantation in the HIV-infected patient: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Urinary tract infections in solid organ transplant recipients: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Interactions between anti-infective agents and immunosuppressants—Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Strategies for safe living following solid organ transplantation—Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Travel medicine, transplant tourism, and the solid organ transplant recipient—Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Pneumonia in solid organ transplantation: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Donor-derived infections: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Screening of donor and candidate prior to solid organ transplantation—Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Diagnosis and management of diarrhea in solid-organ transplant recipients: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Vaccination of solid organ transplant candidates and recipients: Guidelines from the American society of transplantation infectious diseases community of practice",
"Clinical practice guidelines standardisation of immunosuppressive and anti-infective drug regimens in UK paediatric renal transplantation: the harmonisation programme",
"Renal association clinical practice guideline in post-operative care in the kidney transplant recipient",
"KHA-CARI guideline: KHA-CARI adaptation of the KDIGO Clinical Practice Guideline for the Care of Kidney Transplant Recipients",
"Infection in solid-organ transplant recipients",
"Infection in organ transplantation: risk factors and evolving patterns of infection",
"Infection in Organ Transplantation",
"Impact of solid organ transplantation and immunosuppression on fever, leukocytosis, and physiologic response during bacterial and fungal infections",
"Evaluation of a Novel Global Immunity Assay to Predict Infection in Organ Transplant Recipients",
"Transmission of infection with human allografts: essential considerations in donor screening",
"Diagnostic and management strategies for donor-derived infections",
"Donor-derived infection--the challenge for transplant safety",
"Transmission of lymphocytic choriomeningitis virus by organ transplantation",
"Infectious complications of antilymphocyte therapies in solid organ transplantation",
"Immunosuppression Modifications Based on an Immune Response Assay: Results of a Randomized, Controlled Trial",
"Immunosuppressive Agents and Infectious Risk in Transplantation: Managing the 'Net State of Immunosuppression'",
"American Society of Transplantation recommendations for screening, monitoring and reporting of infectious complications in immunosuppression trials in recipients of organ transplantation",
"Nocardia infections in solid organ transplantation: Guidelines from the Infectious Diseases Community of Practice of the American Society of Transplantation",
"Mycobacterium tuberculosis infections in solid organ transplantation: Guidelines from the infectious diseases community of practice of the American Society of Transplantation",
"Vancomycin-resistant Enterococcus in solid organ transplant recipients: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Ventricular assist device-related infections and solid organ transplantation—Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Management of Clostridioides (formerly Clostridium) difficile infection (CDI) in solid organ transplant recipients: Guidelines from the American Society of Transplantation Community of Practice",
"Management of infections due to nontuberculous mycobacteria in solid organ transplant recipients—Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Surgical site infections: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Human papillomavirus infection in solid organ transplant recipients: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Multidrug-resistant Gram-negative bacterial infections in solid organ transplant recipients—Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Intra-abdominal infections in solid organ transplant recipients: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Methicillin-resistant Staphylococcus aureus in solid organ transplantation—Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Clinical presentation and outcome of tuberculosis in kidney, liver, and heart transplant recipients in Spain. Spanish Transplantation Infection Study Group, GESITRA",
"Prevalence of Clostridium difficile infection among solid organ transplant recipients: a meta-analysis of published studies",
"Tuberculosis in solid-organ transplant recipients: consensus statement of the group for the study of infection in transplant recipients (GESITRA) of the Spanish Society of Infectious Diseases and Clinical Microbiology",
"Outcome of Transplantation Using Organs From Donors Infected or Colonized With Carbapenem-Resistant Gram-Negative Bacteria",
"Vancomycin-resistant Enterococcus in liver transplantation: what have we left behind?",
"Is bacteremic sepsis associated with higher mortality in transplant recipients than in nontransplant patients? A matched case-control propensity-adjusted study",
"RNA respiratory viral infections in solid organ transplant recipients: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Cytomegalovirus in solid organ transplant recipients—Guidelines of the American Society of Transplantation Infectious Diseases Community of Practice",
"Viral hepatitis: Guidelines by the American Society of Transplantation Infectious Disease Community of Practice",
"Human herpesvirus 6, 7, and 8 in solid organ transplantation: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Herpes simplex virus infections in solid organ transplantation: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Adenovirus in solid organ transplant recipients: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"BK polyomavirus in solid organ transplantation—Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Human parvovirus B19 in solid organ transplantation: Guidelines from the American society of transplantation infectious diseases community of practice",
"Human T-cell lymphotrophic virus in solid-organ transplant recipients: Guidelines from the American society of transplantation infectious diseases community of practice",
"Arenaviruses and West Nile Virus in solid organ transplant recipients: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Varicella zoster virus in solid organ transplantation: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Post-transplant lymphoproliferative disorders, Epstein-Barr virus infection, and disease in solid organ transplantation: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"KHA-CARI guideline: Cytomegalovirus disease and kidney transplantation",
"COVID-19 in solid organ transplant recipients: Initial report from the US epicenter",
"Incidence and outcome of SARS-CoV-2 infection on solid organ transplantation recipients: A nationwide population-based study",
"COVID-19 in solid organ transplant recipients: Dynamics of disease progression and inflammatory markers in ICU and non-ICU admitted patients",
"Novel Coronavirus-19 (COVID-19) in the immunocompromised transplant recipient: #Flatteningthecurve",
"Transmission of rabies virus from an organ donor to four transplant recipients",
"PHS guideline for reducing human immunodeficiency virus, hepatitis B virus, and hepatitis C virus transmission through organ transplantation",
"A new arenavirus in a cluster of fatal transplant-associated diseases",
"Probability of viremia with HBV, HCV, HIV, and HTLV among tissue donors in the United States",
"Twelve-Month Outcomes After Transplant of Hepatitis C-Infected Kidneys Into Uninfected Recipients: A Single-Group Trial",
"Trial of Transplantation of HCV-Infected Kidneys into Uninfected Recipients",
"Perioperative Ledipasvir-Sofosbuvir for HCV in Liver-Transplant Recipients",
"Heart and Lung Transplants from HCV-Infected Donors to Uninfected Recipients",
"Early outcomes using hepatitis C-positive donors for cardiac transplantation in the era of effective direct-acting anti-viral therapies",
"Delayed seroconversion and rapid onset of lymphoproliferative disease after transmission of human T-cell lymphotropic virus type 1 from a multiorgan donor",
"The independent role of cytomegalovirus as a risk factor for invasive fungal disease in orthotopic liver transplant recipients. Boston Center for Liver Transplantation CMVIG-Study Group. Cytogam, MedImmune, Inc. Gaithersburg, Maryland",
"Quantification of Torque Teno Virus Viremia as a Prospective Biomarker for Infectious Disease in Kidney Allograft Recipients",
"Cytomegalovirus-specific T-cell responses and viral replication in kidney transplant recipients",
"Chronic norovirus infection after kidney transplantation: molecular evidence for immune-driven viral evolution",
"Emerging fungal infections in solid organ transplant recipients: Guidelines of the American Society of Transplantation Infectious Diseases Community of Practice",
"Cryptococcosis in solid organ transplantation—Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Invasive Aspergillosis in solid-organ transplant recipients: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Endemic fungal infections in solid organ transplant recipients—Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Pneumocystis jiroveci in solid organ transplantation: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Candida infections in solid organ transplantation: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Invasive fungal infections among organ transplant recipients: results of the Transplant-Associated Infection Surveillance Network (TRANSNET)",
"Microsporidiosis acquired through solid organ transplantation: a public health investigation",
"Cryptosporidium enteritis in solid organ transplant recipients: multicenter retrospective evaluation of 10 cases reveals an association with elevated tacrolimus concentrations",
"Tissue and blood protozoa including toxoplasmosis, Chagas disease, leishmaniasis, Babesia, Acanthamoeba, Balamuthia, and Naegleria in solid organ transplant recipients— Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Intestinal parasites including Cryptosporidium, Cyclospora, Giardia, and Microsporidia, Entamoeba histolytica, Strongyloides, Schistosomiasis, and Echinococcus: Guidelines from the American Society of Transplantation Infectious Diseases Community of Practice",
"Strongyloidiasis in transplant patients",
"Leishmaniasis among organ transplant recipients",
"Notes from the field: transplant-transmitted Balamuthia mandrillaris --- Arizona, 2010",
"Transmission of Balamuthia mandrillaris by Organ Transplantation",
"Heart transplantation for chronic Chagas' heart disease",
"Risk factors, clinical features, and outcomes of toxoplasmosis in solid-organ transplant recipients: a matched case-control study"
    ]

    for pub in publications:
        st.markdown(f"- {pub}")

# Footer
st.markdown("---")
st.markdown("<h5 style='text-align: center;'>🚀 Developed with OpenAI GPT-4o, Meta LLaMA, and DeepSeek R1</h5>", unsafe_allow_html=True)
st.markdown("""This AI-powered application uses advanced **retrieval-augmented generation (RAG)** language model to extract insights from scientific literature and answer questions in transplant infections. The model generates three responses to each query from **OpenAI GPT-4o, Facebook LLaMA 3.2, and DeepSeek R1**, and synthesizes a final response from those answers. The model may use info outside the document to enhance the answer, and when it does, it will mention that. 

- **App Developed By:** Shreyas Joshi  
- **Literature Corpus Curated By:** Frank Liu, Berk Maden, and Shreyas Joshi  
- **Institution:** Keating Lab, NYU Langone Health
""", unsafe_allow_html=True)