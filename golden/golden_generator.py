import os, re, glob, json, random
from dotenv import load_dotenv

load_dotenv()

# Prevent premature timeout on OpenAI API calls during generation
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"

from deepeval.synthesizer import Synthesizer
from deepeval.models.llms.openai_model import OpenAIModel
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


# Workaround for gpt-4o-mini structured outputs bug:
# Standard JSON mode prevents gpt-4o-mini from entering 16k token runaway loops
class SafeOpenAIModel(OpenAIModel):
    def supports_structured_outputs(self) -> bool:
        return False


# --- Clean VTT transcripts and preserve session metadata ---
def load_chunks_with_metadata():
    chunk_items = []
    skip_phrases = ["hi guys", "good evening", "good night", "bye", "repeat session", "can you hear me"]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for path in sorted(glob.glob("data/*.vtt")):
        session_match = re.search(r"Session[ _]*(\d+)", path)
        session = session_match.group(1) if session_match else "unknown"

        lines = []
        with open(path) as f:
            for ln in f:
                s = ln.strip()
                if not s or s == "WEBVTT" or "-->" in s:
                    continue
                if any(phrase in s.lower() for phrase in skip_phrases):
                    continue
                lines.append(s)

        doc_text = " ".join(lines)
        doc_chunks = splitter.split_text(doc_text)
        for c in doc_chunks:
            # Keep substantive chunks only (>350 chars)
            if len(c.strip()) > 350:
                chunk_items.append({"text": c.strip(), "session": session})

    return chunk_items


if __name__ == "__main__":
    items = load_chunks_with_metadata()
    print(f"📚 Loaded {len(items)} substantive chunks across sessions.")

    # Start with 5 chunks to test; increase to 10 or 15 as needed
    sample_items = random.sample(items, min(5, len(items)))
    contexts = [[item["text"]] for item in sample_items]

    safe_model = SafeOpenAIModel("gpt-4o-mini")
    synthesizer = Synthesizer(
        model=safe_model,
        max_concurrent=2,
    )

    print("✨ Generating goldens with Synthesizer...")
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=1,
    )

    # Convert to standard schema with auto-populated session metadata
    rows = []
    for i, (g, item) in enumerate(zip(goldens, sample_items), 1):
        rows.append({
            "id": f"g{i:03d}",
            "query": g.input,
            "ideal_answer": g.expected_output,
            "source": f"Session {item['session']}",
        })

    output_file = "golden/retriever_deepeval_goldens.json"
    with open(output_file, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"✅ Successfully wrote {len(rows)} goldens -> {output_file}")
    print("💡 Review generated questions & answers to check grounding and accuracy.")