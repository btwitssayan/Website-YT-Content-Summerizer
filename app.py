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
You are an AI assistant specializing in analyzing and summarizing research papers in the field of biomedical models. Your task is to provide a concise and accurate summary of the given content in 300 words or less.

Content to summarize:
{text}

Provide the summary in the following format:

Name of the Model:
Purpose of the Model:
Production-level Feasibility: (Can the model be effectively deployed at scale?)
Dataset Used for Training:
Performance Metrics: (Key performance indicators such as accuracy, F1-score, AUC, etc.)
Code Snippet for Testing: (Include a minimal code example for testing the model, if applicable)
How to Finetune farther:
Ensure clarity, relevance, and technical precision in the response.
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
