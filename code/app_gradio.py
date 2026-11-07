"""Optional web page:  python code/app_gradio.py   then open http://127.0.0.1:7860"""
import gradio as gr

from rag_pipeline import SanskritRAG

rag = SanskritRAG()


def respond(question):
    res = rag.ask(question)
    sources = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(res["chunks"], start=1))
    t = res["timings"]
    stats = f"retrieval {t['retrieval_s']}s | generation {t['generation_s']}s"
    return res["query_devanagari"], sources, res["answer"], stats


demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(label="Question (Sanskrit or Roman letters)"),
    outputs=[
        gr.Textbox(label="Query in Devanagari"),
        gr.Textbox(label="Retrieved passages", lines=8),
        gr.Textbox(label="Answer", lines=4),
        gr.Textbox(label="Timing"),
    ],
    title="Sanskrit RAG",
)

if __name__ == "__main__":
    demo.launch()