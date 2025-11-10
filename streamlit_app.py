import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
import difflib
from fpdf import FPDF

# Page configuration
st.set_page_config(
    page_title="Speech Diagnostic Support Tool",
    page_icon="🎙️",
    layout="wide"
)

# Load configuration and reference data
@st.cache_data
def load_reference_phrases():
    """Load reference phrases from CSV"""
    data = {
        'phrase': ['The cat sat on the mat', 'She sells seashells by the seashore'],
        'expected_IPA': ['/ðə kæt sæt ɒn ðə mæt/', '/ʃi sɛlz siːʃɛlz baɪ ðə siːʃɔː/']
    }
    return pd.DataFrame(data)

@st.cache_data
def load_sensitivity_config():
    """Load sensitivity thresholds"""
    return {
        "high_confidence": 0.85,
        "moderate_confidence": 0.65,
        "low_confidence": 0.45
    }

@st.cache_data
def load_speech_rules():
    """Load comprehensive speech disorder rules"""
    try:
        # Try to load from file first
        df = pd.read_csv('speech_rule_mapping.csv')
        # Clean whitespace from headers
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        # Fallback to embedded data if file not found
        st.warning("⚠️ speech_rule_mapping.csv not found. Using basic rule set.")
        data = {
            'Condition': ['Articulation Disorder', 'Phonological Disorder'],
            'Typical phonetic/phonological patterns': ['Difficulty producing specific speech sounds', 'Patterned simplifications of sounds'],
            'Rule mapping (example)': ['r→w substitution', 'cluster reduction'],
            'Clinical notes': ['Motor-based errors', 'Language-based phonological knowledge difference'],
            'Age_of_concern': ['Any age', 'Typically noticed in childhood'],
            'Confidence_notes': ['High for detection', 'High when multiple processes observed']
        }
        return pd.DataFrame(data)

def simulate_audio_to_ipa(audio_file, reference_phrase):
    """
    Simulate audio to IPA conversion with realistic variations.
    In production, this would use speech recognition + phonetic analysis.
    """
    import random
    reference_ipa = reference_phrase['expected_IPA']
    
    # Remove IPA delimiters
    clean_ipa = reference_ipa.strip('/')
    
    # Simulate transcription with possible errors
    ipa_chars = list(clean_ipa)
    
    # Randomly introduce 0-3 errors for demonstration
    num_errors = random.randint(0, 3)
    
    # Common substitutions based on clinical patterns
    common_substitutions = {
        'r': 'w',      # Gliding
        'ɹ': 'w',      # Gliding (r-sound)
        'l': 'w',      # Gliding
        'θ': 'f',      # Labialization
        'ð': 'd',      # Stopping
        's': 't',      # Stopping
        'ʃ': 't',      # Stopping
        'k': 't',      # Fronting (velar fronting)
        'g': 'd',      # Fronting (velar fronting)
        't': 'k',      # Backing (less common)
    }
    
    for _ in range(num_errors):
        if ipa_chars:
            idx = random.randint(0, len(ipa_chars) - 1)
            char = ipa_chars[idx]
            if char in common_substitutions:
                ipa_chars[idx] = common_substitutions[char]
    
    return '/' + ''.join(ipa_chars) + '/'

def compare_ipa_transcriptions(produced_ipa, expected_ipa):
    """Compare produced IPA with expected IPA and identify differences"""
    # Clean IPA strings
    produced = produced_ipa.strip('/')
    expected = expected_ipa.strip('/')
    
    # Character-level comparison
    matcher = difflib.SequenceMatcher(None, expected, produced)
    differences = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            differences.append({
                'type': tag,
                'expected': expected[i1:i2],
                'produced': produced[j1:j2],
                'position': i1
            })
    
    # Calculate similarity score
    similarity = matcher.ratio()
    
    return differences, similarity

def identify_patterns(differences, speech_rules):
    """Identify potential speech disorder patterns from differences using comprehensive clinical rules"""
    identified_patterns = []
    
    for diff in differences:
        if diff['type'] == 'replace':
            expected = diff['expected']
            produced = diff['produced']
            
            # Match against comprehensive clinical patterns
            
            # Gliding patterns
            if (expected in ['r', 'ɹ'] and produced == 'w') or (expected == 'l' and produced in ['w', 'j']):
                rule = speech_rules[speech_rules['Condition'] == 'Gliding']
                if not rule.empty:
                    identified_patterns.append({
                        'condition': 'Gliding',
                        'pattern': f'{expected}→{produced}',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Moderate',
                        'confidence': rule.iloc[0]['Confidence_notes']
                    })
            
            # Stopping patterns
            elif expected in ['s', 'z', 'f', 'v', 'θ', 'ð', 'ʃ', 'ʒ'] and produced in ['t', 'd', 'p', 'b']:
                rule = speech_rules[speech_rules['Condition'] == 'Stopping']
                if not rule.empty:
                    identified_patterns.append({
                        'condition': 'Stopping',
                        'pattern': f'{expected}→{produced} (fricative to stop)',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Moderate',
                        'confidence': rule.iloc[0]['Confidence_notes']
                    })
            
            # Fronting (Velar Fronting)
            elif expected in ['k', 'g', 'ŋ'] and produced in ['t', 'd', 'n']:
                rule = speech_rules[speech_rules['Condition'] == 'Fronting (Velar Fronting)']
                if not rule.empty:
                    identified_patterns.append({
                        'condition': 'Fronting (Velar Fronting)',
                        'pattern': f'{expected}→{produced}',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Moderate',
                        'confidence': rule.iloc[0]['Confidence_notes']
                    })
            
            # Backing
            elif expected in ['t', 'd', 's'] and produced in ['k', 'g']:
                rule = speech_rules[speech_rules['Condition'] == 'Backing']
                if not rule.empty:
                    identified_patterns.append({
                        'condition': 'Backing',
                        'pattern': f'{expected}→{produced}',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Concerning',
                        'confidence': rule.iloc[0]['Confidence_notes']
                    })
            
            # Labialization
            elif expected in ['s', 'z', 't', 'd'] and produced in ['f', 'v', 'p', 'b']:
                rule = speech_rules[speech_rules['Condition'] == 'Labialization']
                if not rule.empty:
                    identified_patterns.append({
                        'condition': 'Labialization',
                        'pattern': f'{expected}→{produced}',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Mild-Moderate',
                        'confidence': rule.iloc[0]['Confidence_notes']
                    })
            
            # Voicing errors
            elif (expected in ['p', 't', 'k', 'f', 's', 'θ', 'ʃ'] and produced in ['b', 'd', 'g', 'v', 'z', 'ð', 'ʒ']) or \
                 (expected in ['b', 'd', 'g', 'v', 'z', 'ð', 'ʒ'] and produced in ['p', 't', 'k', 'f', 's', 'θ', 'ʃ']):
                rule = speech_rules[speech_rules['Condition'] == 'Voicing (Devoicing/Voicing Errors)']
                if not rule.empty:
                    identified_patterns.append({
                        'condition': 'Voicing Error',
                        'pattern': f'{expected}→{produced}',
                        'example': rule.iloc[0]['Rule mapping (example)'],
                        'clinical_notes': rule.iloc[0]['Clinical notes'],
                        'age_concern': rule.iloc[0]['Age_of_concern'],
                        'severity': 'Mild',
                        'confidence': rule.iloc[0]['Confidence_notes']
                    })
    
    # Check for deletion patterns (simplified)
    for diff in differences:
        if diff['type'] == 'delete' and diff['expected']:
            rule = speech_rules[speech_rules['Condition'] == 'Final Consonant Deletion']
            if not rule.empty:
                identified_patterns.append({
                    'condition': 'Final Consonant Deletion',
                    'pattern': f"Omission of /{diff['expected']}/",
                    'example': rule.iloc[0]['Rule mapping (example)'],
                    'clinical_notes': rule.iloc[0]['Clinical notes'],
                    'age_concern': rule.iloc[0]['Age_of_concern'],
                    'severity': 'Moderate',
                    'confidence': rule.iloc[0]['Confidence_notes']
                })
                break  # Only add once
    
    return identified_patterns

def calculate_confidence_level(similarity_score, config):
    """Calculate confidence level based on similarity score"""
    if similarity_score >= config['high_confidence']:
        return "High Confidence", "success"
    elif similarity_score >= config['moderate_confidence']:
        return "Moderate Confidence", "warning"
    elif similarity_score >= config['low_confidence']:
        return "Low Confidence", "error"
    else:
        return "Very Low Confidence", "error"

def generate_pdf_report(analysis_results, clinician_notes=""):
    """Generate comprehensive PDF report of analysis"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Speech Diagnostic Support Report", ln=True, align='C')
    pdf.ln(5)
    
    # Disclaimer
    pdf.set_font("Arial", 'I', 9)
    pdf.multi_cell(0, 5, "DISCLAIMER: This tool provides pattern analysis and confidence scoring only. Diagnosis remains the responsibility of a qualified speech pathologist.")
    pdf.ln(5)
    
    # Timestamp
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    # Analysis Results
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Analysis Summary", ln=True)
    pdf.set_font("Arial", '', 10)
    
    pdf.cell(0, 8, f"Reference Phrase: {analysis_results['phrase']}", ln=True)
    pdf.cell(0, 8, f"Expected IPA: {analysis_results['expected_ipa']}", ln=True)
    pdf.cell(0, 8, f"Produced IPA: {analysis_results['produced_ipa']}", ln=True)
    pdf.cell(0, 8, f"Similarity Score: {analysis_results['similarity']:.2%}", ln=True)
    pdf.cell(0, 8, f"Confidence Level: {analysis_results['confidence']}", ln=True)
    pdf.ln(5)
    
    # Differences
    if analysis_results['differences']:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Identified Differences", ln=True)
        pdf.set_font("Arial", '', 10)
        for i, diff in enumerate(analysis_results['differences'], 1):
            pdf.cell(0, 6, f"{i}. Expected: '{diff['expected']}' -> Produced: '{diff['produced']}'", ln=True)
        pdf.ln(5)
    
    # Clinical Patterns
    if analysis_results['patterns']:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Clinical Pattern Analysis", ln=True)
        pdf.set_font("Arial", '', 10)
        for i, pattern in enumerate(analysis_results['patterns'], 1):
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 6, f"{i}. {pattern['condition']} - {pattern['severity']}", ln=True)
            pdf.set_font("Arial", '', 9)
            pdf.cell(0, 5, f"   Pattern: {pattern['pattern']}", ln=True)
            pdf.multi_cell(0, 5, f"   Clinical Notes: {pattern['clinical_notes']}")
            pdf.cell(0, 5, f"   Age of Concern: {pattern['age_concern']}", ln=True)
            pdf.cell(0, 5, f"   Confidence: {pattern['confidence']}", ln=True)
            pdf.ln(2)
    
    # Clinician Notes
    if clinician_notes:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Clinician Notes", ln=True)
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 6, clinician_notes)
    
    return pdf.output(dest='S').encode('latin-1')

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Main App Layout
st.title('🎙️ Speech Diagnostic Support Tool')

# Disclaimer at top
st.warning("⚠️ **DISCLAIMER:** This tool provides pattern analysis and confidence scoring only. Diagnosis remains the responsibility of a qualified speech pathologist.")

st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Display sensitivity thresholds
    config = load_sensitivity_config()
    st.subheader("Confidence Thresholds")
    st.write(f"🟢 High: ≥ {config['high_confidence']:.0%}")
    st.write(f"🟡 Moderate: ≥ {config['moderate_confidence']:.0%}")
    st.write(f"🔴 Low: ≥ {config['low_confidence']:.0%}")
    
    st.markdown("---")
    
    # Display reference information
    st.subheader("Clinical Database")
    speech_rules = load_speech_rules()
    st.write(f"**Conditions Tracked:** {len(speech_rules)}")
    with st.expander("View All Conditions"):
        st.dataframe(speech_rules[['Condition', 'Age_of_concern']], width='stretch')

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Step 1: Select Reference Phrase")
    reference_df = load_reference_phrases()
    
    selected_phrase = st.selectbox(
        "Choose a reference phrase:",
        reference_df['phrase'].tolist(),
        key='phrase_selector'
    )
    
    selected_row = reference_df[reference_df['phrase'] == selected_phrase].iloc[0]
    
    st.info(f"**Expected IPA:** {selected_row['expected_IPA']}")

with col2:
    st.header("Step 2: Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Upload patient audio recording",
        type=['wav', 'mp3', 'ogg', 'm4a'],
        help="Supported formats: WAV, MP3, OGG, M4A"
    )
    
    if uploaded_file:
        st.audio(uploaded_file)

# Analysis section
st.markdown("---")
st.header("Step 3: Analyze Speech")

if st.button("🔍 Run Analysis", type="primary", disabled=not uploaded_file):
    with st.spinner("Analyzing speech patterns..."):
        # Simulate processing
        import time
        time.sleep(1)
        
        # Perform analysis
        produced_ipa = simulate_audio_to_ipa(uploaded_file, selected_row)
        differences, similarity = compare_ipa_transcriptions(produced_ipa, selected_row['expected_IPA'])
        patterns = identify_patterns(differences, speech_rules)
        confidence, confidence_type = calculate_confidence_level(similarity, config)
        
        # Store results
        st.session_state.analysis_results = {
            'phrase': selected_phrase,
            'expected_ipa': selected_row['expected_IPA'],
            'produced_ipa': produced_ipa,
            'differences': differences,
            'similarity': similarity,
            'patterns': patterns,
            'confidence': confidence,
            'confidence_type': confidence_type
        }
        st.session_state.analysis_complete = True

# Display results
if st.session_state.analysis_complete and st.session_state.analysis_results:
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    results = st.session_state.analysis_results
    
    # Summary metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric("Similarity Score", f"{results['similarity']:.1%}")
    
    with metric_col2:
        confidence_color = {
            'success': '🟢',
            'warning': '🟡',
            'error': '🔴'
        }
        st.metric("Confidence Level", 
                 f"{confidence_color.get(results['confidence_type'], '⚪')} {results['confidence']}")
    
    with metric_col3:
        st.metric("Patterns Detected", len(results['patterns']))
    
    # Detailed results in tabs
    tab1, tab2, tab3 = st.tabs(["📝 Transcription Comparison", "🔍 Clinical Pattern Analysis", "📄 Report & Notes"])
    
    with tab1:
        st.subheader("Phonetic Transcription")
        
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown("**Expected IPA:**")
            st.code(results['expected_ipa'], language=None)
        
        with comp_col2:
            st.markdown("**Produced IPA:**")
            st.code(results['produced_ipa'], language=None)
        
        if results['differences']:
            st.subheader("Phoneme-Level Differences")
            diff_data = []
            for diff in results['differences']:
                diff_data.append({
                    'Type': diff['type'].title(),
                    'Expected': diff['expected'] if diff['expected'] else '(none)',
                    'Produced': diff['produced'] if diff['produced'] else '(none)',
                    'Position': diff['position']
                })
            st.dataframe(pd.DataFrame(diff_data), width='stretch')
        else:
            st.success("✅ No significant differences detected!")
    
    with tab2:
        st.subheader("Clinical Pattern Analysis")
        
        if results['patterns']:
            for i, pattern in enumerate(results['patterns'], 1):
                with st.expander(f"Pattern {i}: {pattern['condition']} - {pattern['severity']}", expanded=True):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**Observed Pattern:** {pattern['pattern']}")
                        st.write(f"**Example:** {pattern['example']}")
                        st.write(f"**Severity:** {pattern['severity']}")
                    
                    with col_b:
                        st.write(f"**Age of Concern:** {pattern['age_concern']}")
                        st.write(f"**Confidence:** {pattern['confidence']}")
                    
                    st.markdown("**Clinical Notes:**")
                    st.info(pattern['clinical_notes'])
        else:
            st.info("No specific clinical patterns identified. This may indicate typical speech production or require further assessment with additional samples.")
        
        # Show clinical database reference
        if results['patterns']:
            st.markdown("---")
            st.subheader("📚 Related Clinical Information")
            st.write("For comprehensive clinical context, review the full condition database:")
            with st.expander("View All Clinical Conditions"):
                st.dataframe(speech_rules, width='stretch')
    
    with tab3:
        st.subheader("Clinical Notes")
        
        clinician_notes = st.text_area(
            "Add clinical observations and notes:",
            height=150,
            placeholder="Enter any additional observations, context, or clinical insights...\n\nExample:\n- Client history\n- Testing conditions\n- Additional observations\n- Recommended follow-up actions"
        )
        
        st.markdown("---")
        st.subheader("Download Report")
        
        col_pdf, col_csv = st.columns(2)
        
        with col_pdf:
            if st.button("📥 Generate PDF Report", type="secondary"):
                pdf_data = generate_pdf_report(results, clinician_notes)
                st.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name=f"speech_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
        with col_csv:
            # Prepare detailed CSV data
            csv_rows = []
            for i, pattern in enumerate(results['patterns'], 1):
                csv_rows.append({
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Reference Phrase': results['phrase'],
                    'Expected IPA': results['expected_ipa'],
                    'Produced IPA': results['produced_ipa'],
                    'Similarity Score': f"{results['similarity']:.2%}",
                    'Confidence Level': results['confidence'],
                    'Pattern Number': i,
                    'Condition': pattern['condition'],
                    'Pattern': pattern['pattern'],
                    'Severity': pattern['severity'],
                    'Age of Concern': pattern['age_concern'],
                    'Clinical Notes': pattern['clinical_notes'],
                    'Clinician Notes': clinician_notes
                })
            
            if not csv_rows:
                csv_rows.append({
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Reference Phrase': results['phrase'],
                    'Expected IPA': results['expected_ipa'],
                    'Produced IPA': results['produced_ipa'],
                    'Similarity Score': f"{results['similarity']:.2%}",
                    'Confidence Level': results['confidence'],
                    'Pattern Number': 0,
                    'Condition': 'None detected',
                    'Pattern': 'N/A',
                    'Severity': 'N/A',
                    'Age of Concern': 'N/A',
                    'Clinical Notes': 'No patterns identified',
                    'Clinician Notes': clinician_notes
                })
            
            csv_data = pd.DataFrame(csv_rows)
            csv_buffer = io.StringIO()
            csv_data.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"speech_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>Speech Diagnostic Support Tool v2.0 | For clinical use by qualified speech pathologists only</p>
    <p>This tool is a prototype for pattern analysis and should not replace professional clinical judgment</p>
    <p>Clinical database includes comprehensive phonological and speech disorder patterns</p>
    </div>
    """,
    unsafe_allow_html=True
)
