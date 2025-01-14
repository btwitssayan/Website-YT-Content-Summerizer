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
You are a highly skilled assistant. Summarize and analyze the provided content by evaluating each model with structured insights. Use the following format for each model:

Name of the Model:
(Provide the name or title of the model being analyzed.)

Purpose of the Model:
(Explain the problem this model addresses or the specific task it is designed to solve. Be concise yet detailed.)

Production-level Feasibility:
(Assess whether the model can be effectively deployed at scale in real-world scenarios. Include considerations like computational efficiency, scalability, and robustness.)

Dataset Used for Training:
(Detail the dataset(s) used for training the model, including its size, diversity, and relevance to the task.)

Performance Metrics:
(Highlight the key performance indicators such as accuracy, precision, recall, F1-score, AUC, latency, or any other relevant metric.)

Strengths:
(Outline the advantages of the model, such as high accuracy, adaptability, simplicity, or innovation.)

Limitations:
(Point out the challenges, drawbacks, or potential risks, such as bias in data, overfitting, high computational requirements, or lack of interpretability.)

Unique Features or Innovations:
(Identify what distinguishes this model from others. Highlight any novel approaches, architectures, or techniques implemented in the model.)

Suggested Improvements or Future Directions:
(Provide actionable suggestions or ideas for enhancing the model’s performance, usability, or scalability in the future.)
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
            llm =ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)
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
