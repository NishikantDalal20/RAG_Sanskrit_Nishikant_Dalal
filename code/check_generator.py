import time

from config import load_config
from generator import Generator

gen = Generator(load_config())
chunks = [{"text": "पर्वतस्य शिखरप्रदेशे घण्टाकर्णः नाम राक्षसः प्रतिवसति ।"}]
start = time.perf_counter()
out = gen.generate("घण्टाकर्णः कः?", chunks)
print(out["text"])
print(f"{out['completion_tokens']} tokens in {time.perf_counter() - start:.1f}s")