# %%
import pytube
import requests
import re
import gradio as gr
from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
import tiktoken

# %%
def get_youtube_description(url: str):
    full_html = requests.get(url).text
    y = re.search(r'shortDescription":"', full_html)
    desc = ""
    count = y.start() + 19  # adding the length of the 'shortDescription":"
    while True:
        # get the letter at current index in text
        letter = full_html[count]
        if letter == "\"":
            if full_html[count - 1] == "\\":
                # this is case where the letter before is a backslash, meaning it is not real end of description
                desc += letter
                count += 1
            else:
                break
        else:
            desc += letter
            count += 1
    return desc

def get_youtube_info(url: str):
    yt = pytube.YouTube(url)
    title = yt.title
    if title is None:
        title = "None"
    desc = get_youtube_description(url)
    if desc is None:
        desc = "None"
    return title, desc

def get_youtube_transcript_loader_langchain(url: str):
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=True
    )
    return loader.load()

def wrap_docs_to_string(docs):
    return " ".join([doc.page_content for doc in docs]).strip()

def get_text_splitter(chunk_size: int, overlap_size: int):
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=overlap_size)

def get_youtube_transcription(url: str):
    text = wrap_docs_to_string(get_youtube_transcript_loader_langchain(url))
    enc = tiktoken.encoding_for_model("gpt-4")
    count = len(enc.encode(text))
    return text, count

def get_transcription_summary(url: str, temperature: float, chunk_size: int, overlap_size: int):
    docs = get_youtube_transcript_loader_langchain(url)
    text_splitter = get_text_splitter(chunk_size=chunk_size, overlap_size=overlap_size)
    split_docs = text_splitter.split_documents(docs)
    llm = Ollama(
        model="llama3",
        base_url="http://localhost:11434",
        temperature=temperature,
    )
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    output = chain.invoke(split_docs)
    return output['output_text']

# # %%
# try:
#     demo.close()
# except:
#     pass


with gr.Blocks() as demo:
    gr.Markdown("""# YouTube Summarizer with Llama 3
                """)
    with gr.Row(equal_height=True) as r0:
        with gr.Column(scale=4) as r0c1:
            url = gr.Textbox(label='YouTube URL', value="https://youtu.be/bvPDQ4-0LAQ")
        with gr.Column(scale=1) as r0c2:
            bttn_info_get = gr.Button('Get Info', variant='primary')
            bttn_clear = gr.ClearButton(interactive=True, variant='stop')
            
    with gr.Row(variant='panel') as r1:
        with gr.Column(scale=2) as r1c1:
            title = gr.Textbox(label='Title', lines=2, max_lines=10, show_copy_button=True)
        with gr.Column(scale=3, ) as r1c2:
            desc = gr.Textbox(label='Description', max_lines=10, autoscroll=False, show_copy_button=True)
            bttn_info_get.click(fn=get_youtube_info,
                                inputs=url,
                                outputs=[title, desc],
                                api_name="get_youtube_info")

    with gr.Row(equal_height=True) as r2:        
        with gr.Column() as r2c1:
            bttn_trns_get = gr.Button("Get Transcription", variant='primary')
            tkncount = gr.Number(label='Token Count (est)')
        with gr.Column() as r2c3:
            bttn_summ_get = gr.Button("Summarize", variant='primary')
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    temperature = gr.Number(label='Temperature', minimum=0.0, step=0.01, precision=-2)
                with gr.Column(scale=1, min_width=100):
                    chunk = gr.Number(label='Chunk Size', minimum=200, step=100, value=4000)
                with gr.Column(scale=1, min_width=100):
                    overlap = gr.Number(label='Overlap Size', minimum=0, step=10, value=0)
        
    with gr.Row() as r3:
        with gr.Column() as r3c1:
            trns_raw = gr.Textbox(label='Transcript', show_copy_button=True)
        with gr.Column() as r3c2:
            trns_sum = gr.Textbox(label="Summary", show_copy_button=True)
    
    bttn_trns_get.click(fn=get_youtube_transcription,
                            inputs=url,
                            outputs=[trns_raw, tkncount]
                            )
    bttn_summ_get.click(fn=get_transcription_summary,
                                inputs=[url, temperature, chunk, overlap],
                                outputs=trns_sum)
    
    bttn_clear.add([url, title, desc, trns_raw, trns_sum, tkncount])


if __name__ == "__main__":
    demo.launch(share=True)