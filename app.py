import streamlit as st
import os
from typing import Dict, List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage

# Load API key from file
def load_api_key():
    """Load API key from file"""
    try:
        with open('/Users/chethan/Documents/era/geminikey.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error("API key file not found at /Users/chethan/Documents/era/geminikey.txt")
        return None
    except Exception as e:
        st.error(f"Error reading API key: {str(e)}")
        return None

# Initialize session state
if 'rules_context' not in st.session_state:
    st.session_state.rules_context = []

class ReviewState(TypedDict):
    text: str
    rules: List[str]
    results: Dict[str, str]
    analysis_complete: bool

def initialize_llm(api_key: str):
    """Initialize the Gemini LLM"""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )

def analyze_rule_compliance(state: ReviewState) -> ReviewState:
    """Analyze if text meets the given rules"""
    api_key = load_api_key()
    if not api_key:
        return {**state, "results": {"error": "API key not available"}, "analysis_complete": True}
    
    llm = initialize_llm(api_key)
    
    text = state["text"]
    rules = state["rules"]
    results = {}
    
    for i, rule in enumerate(rules):
        prompt = f"""
        Analyze the following text against this rule:
        Rule: {rule}
        Text: {text}
        
        Determine if the rule is:
        1. MET - The text clearly follows the rule
        2. NOT_MET - The text violates the rule (provide the violating content)
        3. NOT_FOUND - The rule cannot be evaluated from the given text (suggest what content should be added)
        
        Respond in this exact format:
        STATUS: [MET/NOT_MET/NOT_FOUND]
        CONTENT: [relevant content or suggestion]
        """
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            results[f"rule_{i+1}"] = response.content
        except Exception as e:
            results[f"rule_{i+1}"] = f"ERROR: {str(e)}"
    
    return {
        **state,
        "results": results,
        "analysis_complete": True
    }

def create_workflow():
    """Create the LangGraph workflow"""
    workflow = StateGraph(ReviewState)
    
    workflow.add_node("analyze", analyze_rule_compliance)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", END)
    
    return workflow.compile()

def parse_result(result: str) -> Dict[str, str]:
    """Parse the LLM result into status and content"""
    lines = result.strip().split('\n')
    status = "ERROR"
    content = "Unable to parse result"
    
    for line in lines:
        if line.startswith("STATUS:"):
            status = line.replace("STATUS:", "").strip()
        elif line.startswith("CONTENT:"):
            content = line.replace("CONTENT:", "").strip()
    
    return {"status": status, "content": content}

# Streamlit UI
st.title("AI Text Rule Compliance Checker")
st.write("Check if your text meets specific rules using Gemini AI with LangGraph workflow")

# Check API key availability
api_key = load_api_key()
if not api_key:
    st.stop()

# Rules input with context storage
st.subheader("Rules")
st.write("Enter rules one per line. Previous rules are stored in context.")

if st.session_state.rules_context:
    st.write("**Previously used rules:**")
    for i, rule in enumerate(st.session_state.rules_context, 1):
        st.write(f"{i}. {rule}")

new_rules = st.text_area("Enter new rules (one per line):", height=150)

# Combine new rules with context
all_rules = []
if st.session_state.rules_context:
    all_rules.extend(st.session_state.rules_context)
if new_rules.strip():
    new_rule_list = [rule.strip() for rule in new_rules.strip().split('\n') if rule.strip()]
    all_rules.extend(new_rule_list)
    # Update context with new rules
    st.session_state.rules_context = list(set(all_rules))  # Remove duplicates

# Text input
st.subheader("Text to Review")
text_input = st.text_area("Enter the text you want to check:", height=200)

# Clear context button
if st.button("Clear Rules Context"):
    st.session_state.rules_context = []
    st.rerun()

# Review button
if st.button("Review Text", type="primary"):
    if not all_rules:
        st.error("Please enter at least one rule")
    elif not text_input.strip():
        st.error("Please enter text to review")
    else:
        try:
            # Create and run workflow
            workflow = create_workflow()
            
            initial_state = {
                "text": text_input,
                "rules": all_rules,
                "results": {},
                "analysis_complete": False
            }
            
            with st.spinner("Analyzing text against rules..."):
                result = workflow.invoke(initial_state)
            
            # Display results
            st.subheader("Review Results")
            
            for i, rule in enumerate(all_rules):
                st.write(f"**Rule {i+1}:** {rule}")
                
                result_key = f"rule_{i+1}"
                if result_key in result["results"]:
                    parsed = parse_result(result["results"][result_key])
                    status = parsed["status"]
                    content = parsed["content"]
                    
                    if status == "MET":
                        st.success(f"✅ **MET**")
                    elif status == "NOT_MET":
                        st.error(f"❌ **NOT MET**")
                        st.write(f"**Violating content:** {content}")
                    elif status == "NOT_FOUND":
                        st.warning(f"⚠️ **NOT FOUND**")
                        st.write(f"**Suggested content:** {content}")
                    else:
                        st.error(f"**Error:** {content}")
                else:
                    st.error("No result available for this rule")
                
                st.divider()
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# Sidebar with instructions
with st.sidebar:
    st.header("How to Use")
    st.write("""
    1. Add rules (one per line)
    2. Enter text to review
    3. Click 'Review Text'
    
    **Rule Context:**
    - Rules are stored in context
    - Add new rules and they'll be combined with previous ones
    - Use 'Clear Rules Context' to reset
    
    **Results:**
    - ✅ MET: Text follows the rule
    - ❌ NOT MET: Shows violating content
    - ⚠️ NOT FOUND: Suggests missing content
    """)
    
    st.header("Example Rules")
    st.code("""
Must include a conclusion
Should have at least 3 paragraphs
Must mention specific benefits
No spelling errors allowed
Include contact information
    """)