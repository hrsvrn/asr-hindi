try:
    import streamlit as st
    import streamlit.components.v1 as components
    from langgraph.graph import StateGraph, END
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from typing import TypedDict, List, Dict
    import google.generativeai as genai
    import numpy as np
    import json
    from inference_wrapped import transcribe_from_mic
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    exit(1)

# Configure Gemini API
import os
api_key = 'AIzaSyBMadNj9xk2AxinNTqnzDbC3EIO_j5i8rE'
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-pro-preview-06-05')

# Define chatbot state
class ChatbotState(TypedDict):
    user_input: str
    history: List[Dict[str, str]]
    response: str
    knowledge_base: List[str]
    is_first_message: bool
    retrieved_context: List[str]

# Load and flatten knowledge base
with open("./wheels_eye_conversations.json", "r", encoding="utf-8") as f:
    kb_raw = json.load(f)

flattened_kb = []
for convo in kb_raw.get("conversations", []):
    for msg in convo.get("messages", []):
        text = msg.get("text") or msg.get("message")
        if text:
            flattened_kb.append(text)

# Initialize embedding model with Streamlit-compatible loading
@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model"""
    try:
        import torch
        import os
        
        # Set environment variables to avoid device conflicts
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Initialize on CPU to avoid meta tensor issues
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        st.stop()

# Load the embedding model
embedding_model = load_embedding_model()

# LangGraph Nodes
def initial_node(state: ChatbotState) -> ChatbotState:
    if state["is_first_message"]:
        greeting = "नमस्ते सर, रवि बात कर रहा हूँ WheelsEye कंपनी से, आपकी गाड़ी लोडिंग की है क्या?"
        state["response"] = greeting
        state["history"].append({"role": "rep", "text": greeting})
        state["is_first_message"] = False
    return state

def retrieve_node(state: ChatbotState) -> ChatbotState:
    if not state["user_input"]:
        return state
    query_embedding = embedding_model.encode(state["user_input"])
    kb_embeddings = embedding_model.encode(state["knowledge_base"])
    scores = cosine_similarity([query_embedding], kb_embeddings)[0]
    top_indices = np.argsort(scores)[-3:][::-1]
    context = [state["knowledge_base"][i] for i in top_indices if scores[i] > 0.3]
    state["retrieved_context"] = context
    return state

def generate_node(state: ChatbotState) -> ChatbotState:
    context_text = "\n".join(state.get("retrieved_context", []))
    history_text = "\n".join([f'{x["role"]}: {x["text"]}' for x in state["history"]])
    prompt = f"""
तुम WheelsEye कंपनी के सेल्स प्रतिनिधि हो (नाम रवि)। 
तुम्हारा काम है ऑपरेटर से जानकारी निकालना:
- गाड़ी लोडिंग वाली है?
- कहाँ से लोड लेते हैं?
- कमीशन लेते हैं?
- रूट और गाड़ी डिटेल क्या है?

तुमको केवल प्रतिनिधि की तरह बोलना है। कोई स्पष्टीकरण या संभावित उत्तर मत दो। 
सिर्फ एक लाइन में संवाद करो।

अब तक की बातचीत:
{history_text}

Knowledge Base:
{context_text}

अब जवाब दो या सवाल पूछो।
"""
    result = model.generate_content(prompt)
    reply = result.text.strip() if result.text else "उत्तर उत्पन्न नहीं हो पाया।"
    state["response"] = reply
    state["history"].append({"role": "rep", "text": reply})
    return state

# LangGraph Setup
chat_graph = StateGraph(ChatbotState)
chat_graph.add_node("initial", initial_node)
chat_graph.add_node("retrieve", retrieve_node)
chat_graph.add_node("generate", generate_node)
chat_graph.set_entry_point("initial")
chat_graph.add_edge("initial", "retrieve")
chat_graph.add_edge("retrieve", "generate")
chat_graph.add_edge("generate", END)
compiled_graph = chat_graph.compile()

# Page Config
st.set_page_config(page_title="WheelsEye Chatbot", page_icon="🚚", layout="centered")

# CSS for styling the chat
st.markdown("""
    <style>
    .chat-box {
        max-height: 450px;
        overflow-y: auto;
        padding: 10px;
        background-color: #000000;
        border: 1px solid #ccc;
        border-radius: 10px;
    }
    .user-msg, .rep-msg {
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
        border-radius: 20px;
        display: inline-block;
    }
    .user-msg {
        background-color: #000000;
        text-align: right;
        float: right;
    }
    .rep-msg {
        background-color: #000000;
        text-align: left;
        float: left;
    }
    .clearfix {
        clear: both;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚚 WheelsEye Chatbot")

# Initialize session state
if "chat_state" not in st.session_state:
    st.session_state.chat_state = {
        "user_input": "",
        "history": [],
        "response": "",
        "knowledge_base": flattened_kb,
        "is_first_message": True,
        "retrieved_context": []
    }
    st.session_state.chat_state = initial_node(st.session_state.chat_state)  # Only show greeting

# Chat box display
st.markdown('<div class="chat-box">', unsafe_allow_html=True)
for msg in st.session_state.chat_state["history"]:
    msg_class = "rep-msg" if msg["role"] == "rep" else "user-msg"
    st.markdown(f'<div class="{msg_class}">{msg["text"]}</div><div class="clearfix"></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input area
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Type your message...", key="user_input_text", label_visibility="collapsed", placeholder="Type your message and hit Send...")
with col2:
    if st.button("Send") and user_input.strip():
        st.session_state.chat_state["user_input"] = user_input.strip()
        st.session_state.chat_state["history"].append({"role": "operator", "text": user_input})
        st.session_state.chat_state = compiled_graph.invoke(st.session_state.chat_state)
        st.rerun()

# 🎙️ Mic input
if st.button("🎙️ Speak (Hindi)"):
    with st.spinner("Listening..."):
        try:
            mic_text = transcribe_from_mic()
            if mic_text:
                st.session_state.chat_state["user_input"] = mic_text.strip()
                st.session_state.chat_state["history"].append({"role": "operator", "text": mic_text.strip()})
                st.session_state.chat_state = compiled_graph.invoke(st.session_state.chat_state)
                st.rerun()
        except Exception as e:
            st.error(f"Voice input failed: {e}")

# Reset chat
if st.button("🔁 Reset Chat"):
    st.session_state.clear()
    st.rerun()
