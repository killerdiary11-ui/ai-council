import streamlit as st
import asyncio
from openai import AsyncOpenAI

# SECURITY UPDATE: Get key from Streamlit Secrets (Server Settings)
# Do not hardcode the key here!
try:
    API_KEY = st.secrets["OPENROUTER_API_KEY"]
except:
    st.error("API Key not found. Please set it in Streamlit Secrets.")
    st.stop()

BASE_URL = "https://openrouter.ai/api/v1"

MODELS = {
    "ChatGPT-4o": "openai/gpt-4o",
    "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
    "Gemini Pro 1.5": "google/gemini-pro-1.5",
    "Perplexity Sonar": "perplexity/sonar-reasoning", 
    "Grok Beta": "xai/grok-beta"
}

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

async def get_ai_response(model_name, model_id, query):
    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be concise."},
                {"role": "user", "content": query}
            ],
        )
        return model_name, response.choices[0].message.content
    except Exception as e:
        return model_name, f"Error: {str(e)}"

async def get_final_conclusion(query, all_responses):
    context_text = ""
    for name, response in all_responses.items():
        context_text += f"\n=== {name} said: ===\n{response}\n"

    final_prompt = f"""
    User Query: "{query}"
    I have asked 5 different AIs this question. Here are their answers:
    {context_text}
    TASK: Analyze these responses. Identify the consensus and write a definitive, final conclusion.
    """
    try:
        response = await client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[{"role": "user", "content": final_prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Could not generate conclusion."

# --- MAIN APP UI ---
st.set_page_config(page_title="Multi-AI Search", layout="wide")
st.title("🤖 The AI Council")
st.markdown("Ask one question. Get answers from **Grok, ChatGPT, Claude, Gemini, and Perplexity**.")

user_query = st.text_input("What do you want to know?")

if st.button("Consult the Council") and user_query:
    results_container = st.container()
    
    async def run_queries():
        tasks = [get_ai_response(name, m_id, user_query) for name, m_id in MODELS.items()]
        results = await asyncio.gather(*tasks)
        return {name: content for name, content in results}

    with st.spinner("All 5 AIs are thinking..."):
        # Create a new event loop for the web server environment
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ai_results = loop.run_until_complete(run_queries())

    st.subheader("Individual Perspectives")
    cols = st.columns(len(ai_results))
    
    for i, (name, content) in enumerate(ai_results.items()):
        with cols[i]:
            st.info(f"**{name}**")
            st.caption(content[:600] + "..." if len(content) > 600 else content)
            with st.expander("Read Full"):
                st.write(content)

    st.divider()
    st.subheader("⚖️ The Final Verdict")
    with st.spinner("Synthesizing final conclusion..."):
        final_verdict = loop.run_until_complete(get_final_conclusion(user_query, ai_results))
        st.success(final_verdict)
