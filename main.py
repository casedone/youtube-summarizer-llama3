from youtube_transcript_api import YouTubeTranscriptApi
import re
from urllib.request import urlopen
import html
import gradio as gr
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
import tiktoken
from tqdm import tqdm

# Prompt templates
MAP_TEMPLATE_TXT = """Write a detail summary of this text section in bullet points. 
Use '-' for bullet points and answer only the bullet points.
Text:
{text}

SUMMARY:"""
    
COMBINE_TEMPLATE_TXT = """Combine these summaries into a final summary in bullet points.
Use '-' for bullet points and answer only the bullet points.
Text:
{text}

FINAL SUMMARY:"""

QUESTION_TEMPLATE_TXT = """Write a detailed summary (in bullet points, using "-" for bullets) of the following:
<text>
{text}
</text>

SUMMARY:"""

REFINE_TEMPLATE_TXT = """Your job is to produce a final summary in bullet points (using "-" for bullets).
You are provided an existing summary here:
<existing_summary>
{existing_answer}
</existing_summary>

You are provided new text.
<new_text>
{text}
</new_text>

Given the new text, refine the original summary.
If the context isn't useful, return the original summary. Answer your summary only, not other texts.
Final Summary:
"""

# Configuration settings
model = "llama3.2"
base_url = "http://localhost:11434"
chunk_size = 2000  # this is in tokens
overlap_size = 0  # this is in tokens
temperature = 0.5
mapreduce_num_predict = 512
map_num_predict = 512  # number of tokens to predict, Default: 128, -1 = infinite generation, -2 = fill context
combine_num_predict = 2048
refine_num_predict = 2048

global config
config = {
    "model": model,
    "base_url": base_url,
    "chunk_size": chunk_size,
    "overlap_size": overlap_size,
    "temperature": temperature,
    "mapreduce_num_predict": mapreduce_num_predict,
    "map_num_predict": map_num_predict,
    "combine_num_predict": combine_num_predict,
    "refine_num_predict": refine_num_predict
}

global text_to_summarize
text_to_summarize = ""


def get_youtube_info(url: str):
    """Get video title and description."""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
        
    video_url = f"https://youtube.com/watch?v={video_id}"
    content = urlopen(video_url).read().decode('utf-8')
    
    title_match = re.search(r'"title":"([^"]+)"', content)
    title = html.unescape(title_match.group(1)) if title_match else "Unknown Title"
    
    desc_match = re.search(r'"description":{"simpleText":"([^"]+)"', content)
    description = html.unescape(desc_match.group(1)) if desc_match else "No description available"
    
    return title, description
    
    
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
    """Get text splitter."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=overlap_size)


def convert_text_to_tokens(text, encoder="gpt-3.5-turbo"):
    """Convert text to tokens."""
    enc = tiktoken.encoding_for_model(encoder)
    return enc.encode(text)


def get_larger_context_size(token_count):
    """Get larger context size."""
    num_ctxes = [1024 * i for i in range(1, 100)]
    num_ctx = next(ctx for ctx in num_ctxes if ctx > token_count)
    return num_ctx


def get_youtube_transcript(url):
    """
    Extract transcript from a YouTube video URL.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str: Full transcript text
    """
    try:
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
            
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ' '.join(entry['text'] for entry in transcript_list)

        enc = tiktoken.encoding_for_model("gpt-4")
        count = len(enc.encode(full_transcript))
        
        return full_transcript, count
        
    except Exception as e:
        return f"Error: {str(e)}", 0
    
    
def get_llm(model: str, base_url: str, temperature: float, num_ctx: int = 2048, num_predict: int = 256):
    """Get LLM."""
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict
    )
    return llm


def convert_text_to_split_docs(text, chunk_size, overlap_size):
    """Convert text to split documents."""
    docs = [Document(page_content=text)]
    text_splitter = get_text_splitter(chunk_size=chunk_size, overlap_size=overlap_size)
    split_docs = text_splitter.split_documents(docs)
    return split_docs
    
    
def get_summary_map_reduce_langchain(text_to_summarize: str, map_prompt_txt: str, combine_prompt_text: str):
    """Get summary using map-reduce method with LangChain."""
    global config
    chunk_size = config["chunk_size"]
    overlap_size = config["overlap_size"]
    model = config["model"]
    base_url = config["base_url"]
    temperature = config["temperature"]
    mapreduce_num_predict = config["mapreduce_num_predict"]
    
    split_docs = convert_text_to_split_docs(text_to_summarize, chunk_size, overlap_size)
    
    tokens = (mapreduce_num_predict + chunk_size)
    num_ctx = get_larger_context_size(tokens)
    llm = get_llm(model, base_url, temperature, num_predict=mapreduce_num_predict, num_ctx=num_ctx)

    map_prompt = PromptTemplate(template=map_prompt_txt, input_variables=["text"])
    combine_prompt = PromptTemplate(template=combine_prompt_text, input_variables=["text"])
    
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt, verbose=True)
    
    output = chain.invoke(split_docs)
    
    return output['output_text']


def get_summary_map_reduce_manual(text_to_summarize: str, map_prompt_txt: str, combine_prompt_text: str):
    """Get summary using map-reduce method manually."""
    global config
    chunk_size = config["chunk_size"]
    overlap_size = config["overlap_size"]
    model = config["model"]
    base_url = config["base_url"]
    temperature = config["temperature"]
    map_num_predict = config["map_num_predict"]
    combine_num_predict = config["combine_num_predict"]
    
    split_docs = convert_text_to_split_docs(text_to_summarize, chunk_size, overlap_size)
    
    map_prompt = PromptTemplate(template=map_prompt_txt, input_variables=["text"])
    combine_prompt = PromptTemplate(template=combine_prompt_text, input_variables=["text"])

    map_num_ctx = get_larger_context_size(map_num_predict + chunk_size)
    llm_map = get_llm(model, base_url, temperature, num_predict=map_num_predict, num_ctx=map_num_ctx)
    
    summaries = []
    for splic_doc in tqdm(split_docs, desc="Mapping..."):
        full_prompt = map_prompt.format_prompt(text=splic_doc.page_content)
        output = llm_map.invoke(full_prompt.text)
        summaries.append(output.content)
    
    combined_summaries = "\n".join(summaries)
    full_prompt = combine_prompt.format_prompt(text=combined_summaries)
    token_counts = len(convert_text_to_tokens(full_prompt.text))
    combine_num_ctx = get_larger_context_size(token_counts + combine_num_predict)
    llm_combine = get_llm(model, base_url, temperature, num_predict=combine_num_predict, num_ctx=combine_num_ctx)
    output_comb = llm_combine.invoke(full_prompt.text)

    return output_comb.content


def get_summary_refine_langchain(text_to_summarize: str, refine_prompt_txt: str, question_prompt_text: str):
    """Get summary using refine method with LangChain."""
    global config
    chunk_size = config["chunk_size"]
    overlap_size = config["overlap_size"]
    model = config["model"]
    base_url = config["base_url"]
    temperature = config["temperature"]
    refine_num_predict = config["refine_num_predict"]
    
    split_docs = convert_text_to_split_docs(text_to_summarize, chunk_size, overlap_size)
    
    refine_prompt = PromptTemplate.from_template(template=refine_prompt_txt)
    question_prompt = PromptTemplate.from_template(template=question_prompt_text)
    num_ctx = get_larger_context_size(int(2 * (refine_num_predict + chunk_size)))
    llm = get_llm(model, base_url, temperature, num_predict=refine_num_predict, num_ctx=num_ctx)
    
    chain = load_summarize_chain(llm, chain_type="refine", question_prompt=question_prompt, refine_prompt=refine_prompt, document_variable_name="text", initial_response_name="existing_answer", verbose=True)
    output_comb = chain.invoke(split_docs)

    return output_comb['output_text']


# Functions to update global variables
def update_model(value):
    global config
    config["model"] = value

def update_base_url(value):
    global config
    config["base_url"] = value

def update_temperature(value):
    global config
    config["temperature"] = value

def update_chunk_size(value):
    global config
    config["chunk_size"] = value

def update_overlap_size(value):
    global config
    config["overlap_size"] = value

def update_mapreduce_num_predict(value):
    global config
    config["mapreduce_num_predict"] = value

def update_map_num_predict(value):
    global config
    config["map_num_predict"] = value

def update_combine_num_predict(value):
    global config
    config["combine_num_predict"] = value

def update_refine_num_predict(value):
    global config
    config["refine_num_predict"] = value

def update_token_count(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    count = len(enc.encode(text))
    return count


with open("sample-text.txt", "r") as f:
    sample_text = f.read()


with gr.Blocks() as demo:
    gr.Markdown("""# YouTube Summarizer by Case Done
- This app will get YouTube info and transcript, and allow you to summarize it.
- It is based on Ollama, Llama 3.2, and LangChain.
- More specifically, we use methods of map-reduce (LangChain and manual versions) and refine (LangChain version).
- Start by providing a valid YouTube URL in the textbox OR paste a text into the text box at the bottom of the first tab.
                """)
    
    with gr.Row():
        with gr.Column(scale=4):
            pass
        with gr.Column(scale=1, min_width=25):
            bttn_clear = gr.ClearButton(interactive=True, variant='stop')
    
    with gr.Tab(label="YouTube") as tab1:
        gr.Markdown("## Input YouTube Link Here:")
        url = gr.Textbox(label='YouTube URL', value="https://youtu.be/bvPDQ4-0LAQ")
        
        gr.Markdown("## YouTube Info")
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                with gr.Accordion("YouTube Information"):
                    title = gr.Textbox(label='Title', lines=2, max_lines=5, show_copy_button=True)
                    desc = gr.Textbox(label='Description', lines=10, max_lines=20, autoscroll=False, show_copy_button=True)
            with gr.Column(scale=1, min_width=25):
                bttn_info_get = gr.Button('Get Info', variant='primary')
        
        gr.Markdown("## Transcript")
        with gr.Row(equal_height=False):              
            with gr.Column(scale=4):
                trns_raw = gr.Textbox(label='Text to Summarize', show_copy_button=True, autoscroll=True, lines=10, max_lines=500, value="", interactive=True)
            with gr.Column(scale=1, min_width=25):
                bttn_trns_get = gr.Button("Get Transcript", variant='primary')
                gr.Markdown("### Or ...")
                buttn_sample_txt = gr.Button("Load Sample Text", variant='secondary')
                tkncount = gr.Number(label='Token Count (~)', interactive=False)
            
            trns_raw.change(fn=update_token_count, inputs=trns_raw, outputs=tkncount)
            trns_raw.value = sample_text[:1000]
            buttn_sample_txt.click(fn=lambda: sample_text, inputs=[], outputs=trns_raw)
        
    with gr.Tab(label="Summarize") as tab2:
        gr.Markdown("## Model Parameters")
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    model = gr.Dropdown(choices=['llama3.2'], value='llama3.2', label='Ollama models', interactive=True)
                    model.change(fn=update_model, inputs=model)
                with gr.Column(scale=1, min_width=100):
                    base_url = gr.Textbox(label='Base URL', value='http://localhost:11434', interactive=True)
                    base_url.change(fn=update_base_url, inputs=base_url)
                with gr.Column(scale=1, min_width=100):
                    temperature = gr.Number(label='Temperature', minimum=0.0, step=0.01, precision=-2)
                    temperature.change(fn=update_temperature, inputs=temperature)

        gr.Markdown("## Text Splitting Parameters")
        with gr.Group():
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    chunk = gr.Number(label='Chunk Size', minimum=200, step=100, value=2000)
                    chunk.change(fn=update_chunk_size, inputs=chunk)
                with gr.Column(scale=1, min_width=100):
                    overlap = gr.Number(label='Overlap Size', minimum=0, step=10, value=0)
                    overlap.change(fn=update_overlap_size, inputs=overlap)

        gr.Markdown("## Approaches")
        with gr.Tab(label="Map-Reduce LangChain") as tab_mrlc:
            with gr.Row():
                with gr.Column(scale=1, min_width=25):
                    mapreduce_num_predict = gr.Number(label='Number of tokens to predict', minimum=8, step=128, value=config["mapreduce_num_predict"], interactive=True)
                    mapreduce_num_predict.change(fn=update_mapreduce_num_predict, inputs=mapreduce_num_predict)
                with gr.Column(scale=4):
                    with gr.Accordion(label="Prompt Templates", open=False):
                        map_prompt_txt_mrlc = gr.Textbox(label="Prompt for the mapping step", value=MAP_TEMPLATE_TXT, lines=10, max_lines=50, show_copy_button=True, interactive=True)
                        combine_prompt_txt_mrlc = gr.Textbox(label="Prompt for the combine step", value=COMBINE_TEMPLATE_TXT, lines=10, max_lines=50, show_copy_button=True, interactive=True)
            with gr.Row():
                with gr.Column(scale=1, min_width=25):
                    bttn_summ_mrlc = gr.Button("Summarize with Map-Reduce LangChain", variant='primary')
            with gr.Row():
                with gr.Column() as r3c2_mrlc:
                    trns_sum_mrlc = gr.Textbox(label="Summary", show_copy_button=True, lines=20, max_lines=100)

        with gr.Tab(label="Map-Reduce Manual") as tab_mrmn:
            with gr.Row():
                with gr.Column(scale=1, min_width=25):
                    map_num_predict = gr.Number(label='Number of tokens to predict at mapping step', minimum=8, step=128, value=config["map_num_predict"], interactive=True)
                    map_num_predict.change(fn=update_map_num_predict, inputs=map_num_predict)
                with gr.Column(scale=1, min_width=25):
                    combine_num_predict = gr.Number(label='Number of tokens to predict at refine step', minimum=8, step=128, value=config["combine_num_predict"], interactive=True)
                    combine_num_predict.change(fn=update_combine_num_predict, inputs=combine_num_predict)
                with gr.Column(scale=4):
                    with gr.Accordion(label="Prompt Templates", open=False):
                        map_prompt_txt_mrmn = gr.Textbox(label="Prompt for the mapping step", value=MAP_TEMPLATE_TXT, lines=10, max_lines=50, show_copy_button=True)
                        combine_prompt_txt_mrmn = gr.Textbox(label="Prompt for the combine step", value=COMBINE_TEMPLATE_TXT, lines=10, max_lines=50, show_copy_button=True)
            with gr.Row():
                with gr.Column(scale=1, min_width=25):
                    bttn_summ_mrmn = gr.Button("Summarize with Map-Reduce Manual", variant='primary')
            with gr.Row():
                with gr.Column() as r3c2_mrmn:
                    trns_sum_mrmn = gr.Textbox(label="Summary", show_copy_button=True, lines=20, max_lines=100)

        with gr.Tab(label="Refine LangChain") as tab_rflc:
            with gr.Row():
                with gr.Column(scale=1, min_width=25):
                    refine_num_predict = gr.Number(label='Number of tokens to predict', minimum=8, step=128, value=config["refine_num_predict"], interactive=True)
                    refine_num_predict.change(fn=update_refine_num_predict, inputs=refine_num_predict)
                with gr.Column(scale=4):
                    with gr.Accordion(label="Prompt Templates", open=False):
                        question_prompt_txt_rl = gr.Textbox(label="Prompt for the each split doc", value=QUESTION_TEMPLATE_TXT, lines=10, max_lines=50, show_copy_button=True, interactive=True)
                        refine_prompt_txt_rl = gr.Textbox(label="Prompt for the refine step", value=REFINE_TEMPLATE_TXT, lines=10, max_lines=50, show_copy_button=True, interactive=True)
            with gr.Row():
                with gr.Column(scale=1, min_width=25):
                    bttn_summ_rflc = gr.Button("Summarize with Refine LangChain", variant='primary')
            with gr.Row():
                with gr.Column() as r3c2_rflc:
                    trns_sum_rflc = gr.Textbox(label="Summary", show_copy_button=True, lines=20, max_lines=100)

    ### events

                    
    bttn_info_get.click(fn=get_youtube_info,
                        inputs=url,
                        outputs=[title, desc],
                        api_name="get_youtube_info"
                        )
        
    bttn_trns_get.click(fn=get_youtube_transcript,
                        inputs=url,
                        outputs=[trns_raw, tkncount]
                        )

    bttn_summ_mrmn.click(fn=get_summary_map_reduce_manual,
                         inputs=[trns_raw, map_prompt_txt_mrmn, combine_prompt_txt_mrmn],
                         outputs=trns_sum_mrmn
                         )

    bttn_summ_mrlc.click(fn=get_summary_map_reduce_langchain,
                         inputs=[trns_raw, map_prompt_txt_mrlc, combine_prompt_txt_mrlc],
                         outputs=trns_sum_mrlc
                         )


    bttn_summ_rflc.click(fn=get_summary_refine_langchain,
                         inputs=[trns_raw, refine_prompt_txt_rl, question_prompt_txt_rl],
                         outputs=trns_sum_rflc
                         )
    
    bttn_clear.add([url, title, desc, trns_raw, trns_sum_mrmn, trns_sum_mrlc, trns_sum_rflc, tkncount])


if __name__ == "__main__":
    demo.launch(share=False)