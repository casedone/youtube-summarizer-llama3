"""
This is a simple app to summarize YouTube videos. 
It is based on LangChain map-reduce method powered by Ollama.

Credits:
Thanks https://medium.com/the-data-perspectives/custom-prompts-for-langchain-chains-a780b490c199 for the refine prompt template.
"""

from youtube_transcript_api import YouTubeTranscriptApi
import re
from urllib.request import urlopen
import html
import gradio as gr
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM as Ollama
import tiktoken

QUESTION_TEMPLATE_TXT = """Write a concise summary of the following:

"{text}"

CONCISE SUMMARY:"""

REFINE_TEMPLATE_TXT = """Your job is to produce a final summary.
We have provided an existing summary up to a certain point: {existing_answer}
We have the opportunity to refine the existing summary (only if needed) with some more context below.
------------
{text}
------------
Given the new context, refine the original summary.
If the context isn't useful, return the original summary."""

MAP_TEMPLATE_TXT = """Write a detail summary of this text section in bullet points.
Text:
{text}

SUMMARY:"""
    
COMBINE_TEMPLATE_TXT = """Combine these summaries into a final summary in bullet points.
Text:
{text}

FINAL SUMMARY:"""

def get_youtube_info(url: str):
    """Get video title and description."""
    # try:
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
        
    # Get video page content
    video_url = f"https://youtube.com/watch?v={video_id}"
    content = urlopen(video_url).read().decode('utf-8')
    
    # Extract title
    title_match = re.search(r'"title":"([^"]+)"', content)
    title = html.unescape(title_match.group(1)) if title_match else "Unknown Title"
    
    # Extract description
    desc_match = re.search(r'"description":{"simpleText":"([^"]+)"', content)
    description = html.unescape(desc_match.group(1)) if desc_match else "No description available"
    
    return title, description
    # except Exception as e:
    #     return {"title": "Error", "description": str(e)}
    
    
def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_text_splitter(chunk_size: int, overlap_size: int):
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=overlap_size)


def get_youtube_transcript(url):
    """
    Extract transcript from a YouTube video URL.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str: Full transcript text
    """
    try:
        # Extract video ID from URL
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
            
        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine transcript pieces
        full_transcript = ' '.join(entry['text'] for entry in transcript_list)

        enc = tiktoken.encoding_for_model("gpt-4")
        count = len(enc.encode(full_transcript))
        
        return full_transcript, count
        
    except Exception as e:
        return f"Error: {str(e)}", 0
    
    
def get_llm(model: str, base_url: str, temperature: float, max_context: int):
    llm = Ollama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=max_context,
        num_predict=256,
    )
    return llm
    
    
def get_transcription_summary(url: str, model: str, base_url: str, temperature: float, 
                            max_context: int, chunk_size: int, overlap_size: int,
                            map_prompt_txt: str, combine_prompt_text: str, chain_type: str,
                            question_prompt_txt: str, refine_prompt_txt: str):
    
    transcript, tokencount = get_youtube_transcript(url)
    docs = [Document(
        page_content=transcript,
        metadata={"source": url}
    )]

    text_splitter = get_text_splitter(chunk_size=chunk_size, overlap_size=overlap_size)
    split_docs = text_splitter.split_documents(docs)
    
    llm = get_llm(
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_context=max_context
    )

    map_prompt = PromptTemplate(
        template=map_prompt_txt,
        input_variables=["text"]
    )

    combine_prompt = PromptTemplate(
        template=combine_prompt_text,
        input_variables=["text"]
    )
    
    if chain_type == "refine":
        question_prompt = PromptTemplate(
            template=question_prompt_txt,
            input_variables=["text"]
        )
        refine_prompt = PromptTemplate(
            template=refine_prompt_txt,
            input_variables=["existing_answer", "text"]
        )
        chain = load_summarize_chain(
            llm, 
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
            verbose=True
        )
    if chain_type == "map_reduce":
        chain = load_summarize_chain(
            llm, 
            chain_type=chain_type,
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True
        )
    
    output = chain.invoke(split_docs)
    
    return output['output_text']


with gr.Blocks() as demo:
    gr.Markdown("""# YouTube Summarizer by Case Done
- This app will get YouTube info and transcript, and allow you to summarize it.
- It is based on LangChain map-reduce method powered by Llama 3.2 via Ollama.
- Start by providing a valid YouTube URL in the textbox.
                """)
    
    with gr.Row():
        with gr.Column(scale=4):
            pass
        with gr.Column(scale=1, min_width=25):
            bttn_clear = gr.ClearButton(interactive=True, variant='stop')
    
    with gr.Tab(label="YouTube") as tab1:
        
        gr.Markdown("## Input YouTube Link Here:")
        url = gr.Textbox(label='YouTube URL', value="https://www.youtube.com/watch?v=cdiD-9MMpb0&t=1s&ab_channel=LexFridman")
        
        gr.Markdown("## YouTube Info")
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                with gr.Accordion("YouTube Information"):
                    title = gr.Textbox(label='Title', lines=2, max_lines=5, show_copy_button=True)
                    desc = gr.Textbox(label='Description', lines=10, max_lines=20, 
                                      autoscroll=False, show_copy_button=True)
            with gr.Column(scale=1, min_width=25):
                bttn_info_get = gr.Button('Get Info', variant='primary', )
        
        gr.Markdown("## Transcript")
        with gr.Row(equal_height=False):              
            with gr.Column(scale=4):
                trns_raw = gr.Textbox(label='Transcript', show_copy_button=True, autoscroll=True,
                                      lines=10, max_lines=500)
            with gr.Column(scale=1, min_width=25):
                bttn_trns_get = gr.Button("Get Transcript", variant='primary')
                tkncount = gr.Number(label='Token Count (~)', interactive=False)
        
    with gr.Tab(label="Summarize") as tab2:
        gr.Markdown("## Parameters")

        with gr.Accordion(label='Ollama Model Parameters', open=True):
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    model = gr.Textbox(label='Model', value="llama3.2")
                with gr.Column(scale=1, min_width=100):
                    base_url = gr.Textbox(label='Base URL', value="http://localhost:11434")
                with gr.Column(scale=1, min_width=100):
                    temperature = gr.Number(label='Temperature', minimum=0.0, step=0.01, precision=-2)
                with gr.Column(scale=1, min_width=100):
                    max_context = gr.Number(label='Max Context Length', minimum=2096, step=1000, value=4096)

        with gr.Accordion(label='Text Splitting Parameters', open=True):
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    chunk = gr.Number(label='Chunk Size', minimum=200, step=100, value=1000)
                with gr.Column(scale=1, min_width=100):
                    overlap = gr.Number(label='Overlap Size', minimum=0, step=10, value=0)
                    
        with gr.Accordion(label="Prompt Templates", open=False):
            with gr.Accordion(label='Map-reduce method prompts', open=False):
                map_prompt_txt = gr.Textbox(label="Prompt for the mapping step", value=MAP_TEMPLATE_TXT,
                                            lines=10, max_lines=50, show_copy_button=True)
                combine_prompt_txt = gr.Textbox(label="Prompt for the combine step", value=COMBINE_TEMPLATE_TXT,
                                            lines=10, max_lines=50, show_copy_button=True)
                
            with gr.Accordion(label='Refine method prompts', open=False):
                question_prompt_txt = gr.Textbox(label="Initial Question Prompt", value=QUESTION_TEMPLATE_TXT,
                                            lines=10, max_lines=50, show_copy_button=True)
                refine_prompt_txt = gr.Textbox(label="Refine Prompt", value=REFINE_TEMPLATE_TXT,
                                            lines=10, max_lines=50, show_copy_button=True)
                
        gr.Markdown("## Run Summarization")
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                    chain_type = gr.Radio(
                        label="Chain Type",
                        choices=["map_reduce", "refine"],
                        value="refine"
                    )
            with gr.Column(scale=1, min_width=25):
                bttn_summ_get = gr.Button("Summarize", variant='primary')
            
        with gr.Row():
            with gr.Column():
                    trns_sum = gr.Textbox(label="Summary", show_copy_button=True)
                    
    bttn_info_get.click(fn=get_youtube_info,
                        inputs=url,
                        outputs=[title, desc],
                        api_name="get_youtube_info"
                        )
        
    bttn_trns_get.click(fn=get_youtube_transcript,
                        inputs=url,
                        outputs=[trns_raw, tkncount]
                        )
    bttn_summ_get.click(fn=get_transcription_summary,
                        inputs=[url, model, base_url, temperature, max_context, 
                               chunk, overlap, map_prompt_txt, combine_prompt_txt, chain_type,
                               question_prompt_txt, refine_prompt_txt],
                        outputs=trns_sum)
    
    bttn_clear.add([url, title, desc, trns_raw, trns_sum, tkncount])


if __name__ == "__main__":
    demo.launch(share=False)

