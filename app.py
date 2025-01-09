import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')


groq_api_key='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'

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API


prompt_template="""
You are an AI assistant specializing in analyzing and comparing biomedical models. Your task is to deliver a comparison document for the provided research papers/models. For each model, summarize and evaluate its key characteristics in 500 words or less.

Content to summarize:
{text}

Provide the analysis in the following format for each model:

Name of the Model:
Purpose of the Model: (What problem does it solve?)
Production-level Feasibility: (Can the model be effectively deployed at scale?)
Dataset Used for Training:
Performance Metrics: (Key indicators such as accuracy, F1-score, AUC, etc.)
Pros: (Highlight the strengths of the model)
Cons: (List the limitations or challenges of the model)
Unique Features or Innovations: (What sets this model apart from others?)
Ensure that the document delivers a clear and objective comparison, emphasizing strengths, weaknesses, and distinguishing factors to assist in making an informed decision.
"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            llm =ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtu.be" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=False,
                                                            language=["en", "id","hi","bn","be"],
                                                            translation="en",)

                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
