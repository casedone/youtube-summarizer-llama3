# YouTube Summarizer By Case Done

![UI example](./docs/UI%20example.png)

> This repo was updated in December 2024, but originally it contains materials that were discussed in ["Beginner to Master Ollama & Build a YouTube Summarizer with Llama 3 and LangChain"](https://www.youtube.com/live/NMObj5tBKX4?utm_source=github&utm_medium=github-readme).

This repo will teach you how to:
- Use LLM local or API via Ollama and again via LangChain
- Use Llama models (Llama 3l2-3B as of Dec 2024)
- Build UI with Gradio
- Use case = "Summarize YouTube"
- Summarizatin using LangChain's map-reduce approach.
- [added] Customize prompts during the map and combine steps.

Specifically, we will first retrieve a transcript of the target YouTube video (directed via a URL), then we will as the Llama model to summarize it.

## Run it
1. Create conda environment with python=3.12
2. Install these modules

```
pip install gradio openai langchain langchain-community youtube_transcript_api tiktoken transformers langchain-ollama
```
3. Serve Ollama if it's not run already:
```shell
ollama serve
```
If you don't know how, consult my [how-to video](https://www.youtube.com/watch?v=NMObj5tBKX4&t=1786s&utm_source=github&utm_medium=github-readme) or Ollama directly.

4. Simply run:
```shell
python main.py
```

## Tools you will use
- `Ollama` to run local LLM API
- `Llama 3.2-3B` from Meta, to use as AI brain. See on Ollama page.
- `Gradio`, to build UI
- `LangChain` as framework for LLM app
- `tiktoken` library to estimate token counts

## Sharing & Crediting

> Feel free to copy and distribute, but we appreciate you giving us credits.


## ⛓️Connect with Us:

👍 Like | 🔗 Share | 📢 Subscribe | 💬 Comments | ❓ Questions

[LinkedIn](www.linkedin.com/company/casedonebyai) <br>
[YouTube](www.youtube.com/@CaseDonebyAI) <br>
[Facebook](www.facebook.com/casedonebyai) <br>
[TikTok](www.tiktok.com/@casedonebyai) <br>
[Github](www.github.com/casedone) <br>
[SubStack](casedonebyai.substack.com)