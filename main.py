import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import json
from datetime import datetime



# Constants
PAGE_TITLE = "PTE Assistant - Multi-Model Comparison"
PAGE_ICON = "🎓"
MODELS = {
    "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 1.5 Flash-8B": "gemini-1.5-flash-8b"
}
TRAINING_DATA_FILE = "training_data.json"

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_responses" not in st.session_state:
    st.session_state.selected_responses = {}
if "feedback_messages" not in st.session_state:
    st.session_state.feedback_messages = {}

def load_training_data():
    """Load existing training data or return empty list if file doesn't exist"""
    try:
        if os.path.exists(TRAINING_DATA_FILE) and os.path.getsize(TRAINING_DATA_FILE) > 0:
            with open(TRAINING_DATA_FILE, 'r') as f:
                return json.load(f)
        return []
    except json.JSONDecodeError:
        st.warning("Found invalid training data file. Creating new one.")
        return []
    except Exception as e:
        st.error(f"Error loading training data: {str(e)}")
        return []

def save_training_data(data):
    """Save data for future training"""
    try:
        # Load existing data
        existing_data = load_training_data()
        
        # Append new data
        existing_data.append(data)
        
        # Save updated data
        with open(TRAINING_DATA_FILE, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        return True
    except Exception as e:
        st.error(f"Error saving training data: {str(e)}")
        return False

def show_feedback_message(idx, message, is_error=False):
    """Show feedback message and store it in session state"""
    st.session_state.feedback_messages[idx] = {
        "message": message,
        "is_error": is_error,
        "timestamp": datetime.now().isoformat()
    }

def display_feedback(idx):
    """Display feedback message if it exists"""
    if idx in st.session_state.feedback_messages:
        feedback = st.session_state.feedback_messages[idx]
        if feedback["is_error"]:
            st.error(feedback["message"])
        else:
            st.success(feedback["message"])

def set_page_config():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

@st.cache_resource
def initialize_rag_components():
    # Initialize Pinecone
    Pinecone(api_key="7f389225-3c39-4cd0-afc2-d3fafc869cd2")
    
    # Initialize vector store
    docsearch = PineconeVectorStore.from_existing_index(
    index_name = "dataset2",
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
)
    
    # Create retriever
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    # Initialize models
    os.environ["GOOGLE_API_KEY"] = "AIzaSyD9yBsVKCHdICXt60SWte5yS4PoyalHPtw"
    llms = {
        model_name: ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0,
            max_output_tokens=800
        ) for model_name, model_id in MODELS.items()
    }
    
    # System prompt remains the same...
    system_prompt = """
    You are an advanced AI assistant specialized in PTE (Pearson Test of English) exam preparation. Your role is to provide expert guidance, explanations, and strategies to help students excel in all aspects of the PTE exam.
Core Responsibilities:

Provide accurate, detailed information about PTE exam structure, scoring, and recent updates.
Offer tailored advice and strategies for each PTE section: Speaking, Writing, Reading, and Listening.
Suggest effective study plans and time management techniques.
Provide constructive feedback on practice responses (when given).

Guidelines for Responses:

Use the following retrieved context to inform your answers: {context}
If the context doesn't provide sufficient information or
If you don't know the answer or are unsure, clearly state this and suggest reliable resources for further information.
Tailor your language complexity to the user's apparent level of understanding.
Be concise yet thorough. Aim for clear, actionable advice.
Use bullet points or numbered lists for step-by-step instructions or multiple tips.

Ethical Considerations:
Topic Limitation: If a question is outside the scope of the PTE exam, kindly inform the user that you are only equipped to address PTE-related topics.
Never provide or encourage cheating methods.
Emphasize the importance of genuine language skill development over exam tricks.
Respect copyright; produce exact questions from official PTE materials.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Create chains for each model
    chains = {}
    for model_name, llm in llms.items():
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chains[model_name] = create_retrieval_chain(retriever, question_answer_chain)
    
    return chains

def display_chat_history():
    for idx, interaction in enumerate(st.session_state.chat_history):
        # Display user message
        st.write("👤 **You:**", interaction["user_input"])
        
        # Create columns for model responses
        cols = st.columns(len(MODELS))
        for col, (model_name, response) in zip(cols, interaction["model_responses"].items()):
            with col:
                st.write(f"🤖 **{model_name}:**")
                st.write(response)
                
                # Add selection button if response hasn't been selected yet
                if idx not in st.session_state.selected_responses:
                    if st.button(f"Select this response", key=f"select_{idx}_{model_name}"):
                        st.session_state.selected_responses[idx] = {
                            "selected_model": model_name,
                            "selected_response": response
                        }
                        if save_training_data({
                            "timestamp": datetime.now().isoformat(),
                            "question": interaction["user_input"],
                            "model_responses": interaction["model_responses"],
                            "selected_model": model_name,
                            "selected_response": response,
                            "custom_response": None
                        }):
                            show_feedback_message(
                                idx,
                                f"✅ Response from {model_name} has been selected and saved successfully! This response will be used to improve future answers."
                            )
                        else:
                            show_feedback_message(
                                idx,
                                "❌ Failed to save the selected response. Please try again.",
                                is_error=True
                            )
        
        # Show custom response input if no response is selected yet
        if idx not in st.session_state.selected_responses:
            st.write("💡 **If none of the responses are satisfactory, provide a custom answer:**")
            custom_response = st.text_area("Custom response:", key=f"custom_{idx}")
            if st.button("Submit custom response", key=f"submit_custom_{idx}"):
                if not custom_response.strip():
                    show_feedback_message(
                        idx,
                        "❌ Please enter a custom response before submitting.",
                        is_error=True
                    )
                else:
                    st.session_state.selected_responses[idx] = {
                        "selected_model": "custom",
                        "selected_response": custom_response
                    }
                    if save_training_data({
                        "timestamp": datetime.now().isoformat(),
                        "question": interaction["user_input"],
                        "model_responses": interaction["model_responses"],
                        "selected_model": "custom",
                        "selected_response": custom_response,
                        "custom_response": custom_response
                    }):
                        show_feedback_message(
                            idx,
                            "✅ Your custom response has been submitted and saved successfully! This will help improve future responses."
                        )
                    else:
                        show_feedback_message(
                            idx,
                            "❌ Failed to save your custom response. Please try again.",
                            is_error=True
                        )
        #else:
            # Display the selected response
            #selected = st.session_state.selected_responses[idx]
           # st.success(f"✅ Selected response from: {selected['selected_model']}")
        
        # Display any feedback messages for this interaction
        display_feedback(idx)
        
        st.divider()

def main():
    set_page_config()

    st.header("PTE Assistant - Multi-Model Comparison 🎓")
    st.subheader("Compare responses and select the best answer")

    # Initialize RAG chains for all models
    rag_chains = initialize_rag_components()

    # Chat input
    user_input = st.chat_input("Your question:")

    if user_input:
        # Container for responses
        model_responses = {}
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get responses from each model
        for i, (model_name, chain) in enumerate(rag_chains.items()):
            status_text.text(f"Getting response from {model_name}...")
            with st.spinner(f"Thinking... ({model_name})"):
                response = chain.invoke({"input": user_input})
                model_responses[model_name] = response["answer"]
            progress_bar.progress((i + 1) / len(MODELS))
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Store the interaction in chat history
        st.session_state.chat_history.append({
            "user_input": user_input,
            "model_responses": model_responses
        })

    # Display chat history with response selection
    display_chat_history()

    
     

if __name__ == "__main__":
    main()
