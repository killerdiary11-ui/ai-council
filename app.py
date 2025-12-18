import streamlit as st
import asyncio
from openai import AsyncOpenAI

# SECURITY: Get key from Streamlit Secrets
try:
    API_KEY = st.secrets["OPENROUTER_API_KEY"]
except:
    st.error("API Key not found. Please set it in Streamlit Secrets.")
    st.stop()

BASE_URL = "https://openrouter.ai/api/v1"

# --- THE FREE COUNCIL ---
# These models are currently free to use on OpenRouter
MODELS = {
    "Gemini 2.0 Flash": "google/gemini-2.0-flash-exp:free",
    "Llama 3.1 8B": "meta-llama/llama-3.1-8b-instruct:free",
    "Llama 3.2 11B": "meta-llama/llama-3.2-11b-vision-instruct:free",
    "Phi-3 Medium": "microsoft/phi-3-medium-128k-instruct:free",
    "Mistral 7B": "mistralai/mistral-7b-instruct:free"
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
        if "Error" not in response:
            context_text += f"\n=== {name} said: ===\n{response}\n"

    final_prompt = f"""
    User Query: "{query}"
    I have asked several AIs this question. Here are their answers:
    {context_text}
    TASK: Analyze these responses. Identify the consensus and write a definitive, final conclusion.
    """
    try:
        # Use Gemini Flash for the conclusion (Smart & Free)
        response = await client.chat.completions.create(
            model="google/gemini-2.0-flash-exp:free",
            messages=[{"role": "user", "content": final_prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate conclusion. Error: {str(e)}"

# --- MAIN APP UI ---
st.set_page_config(page_title="Free AI Search", layout="wide")
st.title("🤖 The (Free) AI Council")
st.markdown("Ask one question. Get answers from **Gemini, Llama, Phi-3, and Mistral** (100% Free).")

user_query = st.text_input("What do you want to know?")

if st.button("Consult the Council") and user_query:
    results_container = st.container()
    
    async def run_queries():
        tasks = [get_ai_response(name, m_id, user_query) for name, m_id in MODELS.items()]
        results = await asyncio.gather(*tasks)
        return {name: content for name, content in results}

    with st.spinner("The Council is thinking..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ai_results = loop.run_until_complete(run_queries())

    st.subheader("Individual Perspectives")
    cols = st.columns(len(ai_results))
    
    for i, (name, content) in enumerate(ai_results.items()):
        with cols[i]:
            if "Error" in content:
                st.error(f"**{name}**")
                st.caption(content)
            else:
                st.success(f"**{name}**")
                st.caption(content[:600] + "..." if len(content) > 600 else content)
                with st.expander("Read Full"):
                    st.write(content)

    st.divider()
    st.subheader("⚖️ The Final Verdict")
    with st.spinner("Synthesizing final conclusion..."):
        final_verdict = loop.run_until_complete(get_final_conclusion(user_query, ai_results))
        st.info(final_verdict)
