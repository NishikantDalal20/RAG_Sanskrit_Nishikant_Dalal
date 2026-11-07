"""Step 4b: the LLM that writes the answer. Runs on CPU through llama.cpp."""
import os

# SYSTEM_PROMPT = (
#     "You answer questions about Sanskrit stories. Use ONLY the numbered context "
#     "passages given by the user. If the answer is not in the passages, say exactly: "
#     "'The answer is not in the documents.' "
#     "Reply in two lines. Line 1: a short answer in simple Sanskrit (Devanagari). "
#     "Line 2: start with 'English:' and give the same answer in English. "
#     "Mention the passage number you used, for example [2]."
# )

SYSTEM_PROMPT = (
    "You answer questions about Sanskrit stories using ONLY the numbered passages given. "
    "Rules: 1) Answer in one or two short English sentences. "
    "2) On a new line write 'Evidence:' followed by the exact Sanskrit sentence, copied from "
    "the passage, that supports the answer. "
    "3) If the passages do not contain the answer, reply only: The answer is not in the documents."
)

EXAMPLE_USER = (
    "Context passages:\n[1] राजा नगरं गच्छति । सः मन्त्रिणा सह वदति ।\n\n"
    "Question: राजा कुत्र गच्छति?\n\nAnswer:"
)
EXAMPLE_ANSWER = "The king goes to the city.\nEvidence: राजा नगरं गच्छति ।"


class Generator:
    def __init__(self, cfg):
        from llama_cpp import Llama

        g = cfg["generation"]
        model_path = cfg["paths"]["model_dir"] / g["filename"]
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run:  python code/download_model.py"
            )
        self.g = g
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=g["n_ctx"],
            n_threads=g["n_threads"] or os.cpu_count(),
            n_gpu_layers=0,  # 0 = never use a GPU
            verbose=False,
        )

    # @staticmethod
    # def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    #     context = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))
    #     user = f"Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    #     return [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": user},
    #     ]

    @staticmethod
    def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
        context = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))
        user = f"Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE_USER},       # one worked example
            {"role": "assistant", "content": EXAMPLE_ANSWER},
            {"role": "user", "content": user},
        ]

    def generate(self, question: str, chunks: list[dict]) -> dict:
        out = self.llm.create_chat_completion(
            messages=self.build_prompt(question, chunks),
            max_tokens=self.g["max_tokens"],
            temperature=self.g["temperature"],
        )
        return {
            "text": out["choices"][0]["message"]["content"].strip(),
            "completion_tokens": out["usage"]["completion_tokens"],
        }