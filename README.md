# YouTube Summarizer with Llama 3

![UI example](./docs/UI%20example.png)

> This repo contains materials that were discissed in ["Beginner to Master Ollama & Build a YouTube Summarizer with Llama 3 and LangChain"](https://www.youtube.com/live/NMObj5tBKX4?utm_source=github&utm_medium=github-readme).

This repo will teach you how to:
- Use LLM local or API via Ollama and again via LangChain
- Use Llama 3-8B model
- Build UI with Gradio
- Use case = "Summarize YouTube video using Llama 3"

Specifically, we will first retrieve a transcript of the target YouTube video (directed via a URL), then we will as Llama 3 to summarize it. We do it this way because Llama 3 only understand text at the moment.

## Run it
Assuming you have the right python environment and other required tools. You can simply run:
```shell
python main.py
```

## Tools you will use
- Ollama to run local LLM API
- `Llama 3` from Meta, to use as AI brain
- `Gradio`, to build UI
- `pytube` a python library for working with YouTube
- `LangChain` as framework for LLM app
- `tiktoken` library to estimate token counts

## Requirements
- You must have Ollama running. Consult my [how-to video](https://www.youtube.com/watch?v=NMObj5tBKX4&t=1786s&utm_source=github&utm_medium=github-readme) or Ollama directly.
- For using this notebook smoothly, we recommend create a python environment based on our provided `requirements.txt`. <br>This can be done by
```shell
pip install -r requirements.txt
```

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