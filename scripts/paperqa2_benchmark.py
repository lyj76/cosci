"""Benchmark runner for PaperQA2 on Osteoarthritis (OA) and Senescence literature.

Evaluates 10 core biological and mechanistic questions across the provided local PDFs:
1. The ganglioside GD3 and its synthase (ST8SIA1) as novel senescence markers associated with osteoarthritis (GeroScience 2025)
2. A ganglioside-based immune checkpoint enables senescent cells to evade immunosurveillance during aging (Nature Aging 2025)
3. Senescent repair memory in chronic disease: Failed resolution as a driver of persistent organ dysfunction (Ageing Res Rev 2025)
"""

import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

dashscope_base = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
os.environ["OPENAI_BASE_URL"] = dashscope_base
os.environ["OPENAI_API_BASE"] = dashscope_base

import litellm
litellm.drop_params = True

from paperqa import Docs, Settings
from paperqa.settings import MultimodalOptions, ParsingSettings

BENCHMARK_QUESTIONS = [
    {
        "id": "Q1",
        "question": "In osteoarthritis cartilage, synovium, and subchondral bone, how do ST8SIA1 expression and GD3 levels change compared to healthy or normal tissues?",
        "target_paper": "The ganglioside GD3 and its synthase (ST8SIA1) as novel sene.pdf",
        "expected_topics": ["increased ST8SIA1", "increased GD3", "correlation with OA grade/Mankin score"],
    },
    {
        "id": "Q2",
        "question": "Which specific cell types in the OA joint exhibit GD3 positivity, and what senescence or catabolic markers co-localize with GD3?",
        "target_paper": "The ganglioside GD3 and its synthase (ST8SIA1) as novel sene.pdf",
        "expected_topics": ["chondrocytes", "synoviocytes", "CDKN2A/p16", "CDKN2B/p15", "SASP factors", "catabolic enzymes"],
    },
    {
        "id": "Q3",
        "question": "What are the observed effects of intra-articular anti-GD3 monoclonal antibody administration in experimental mouse models of OA? Which joint tissues show protection?",
        "target_paper": "The ganglioside GD3 and its synthase (ST8SIA1) as novel sene.pdf",
        "expected_topics": ["subchondral bone protection", "reduction of bone remodeling", "no direct cartilage/synovium protection"],
    },
    {
        "id": "Q4",
        "question": "What is the relationship between GD3, ST8SIA1, and osteoclasts or bone resorption activity in osteoarthritis?",
        "target_paper": "The ganglioside GD3 and its synthase (ST8SIA1) as novel sene.pdf",
        "expected_topics": ["ST8SIA1 in osteoclasts", "promotes osteoclastogenesis", "resorption genes Acp5/Mmp9/Csk", "GD3 knockout fewer osteoclasts"],
    },
    {
        "id": "Q5",
        "question": "What limitations, caveats, or questions regarding GD3 as a therapeutic target versus an indicative biomarker are discussed by Fissoun et al.?",
        "target_paper": "The ganglioside GD3 and its synthase (ST8SIA1) as novel sene.pdf",
        "expected_topics": ["indicative rather than causative in cartilage", "specific effect on bone remodeling", "need for further target validation"],
    },
    {
        "id": "Q6",
        "question": "How do cell-surface gangliosides on senescent cells act as an immune checkpoint to evade immune surveillance?",
        "target_paper": "A ganglioside-based immune checkpoint enables senescent cell.pdf",
        "expected_topics": ["inhibitory checkpoint", "suppression of cytotoxicity", "sialic acid dependent", "immune evasion"],
    },
    {
        "id": "Q7",
        "question": "Which specific inhibitory immunoreceptors on Natural Killer (NK) cells or macrophages bind to GD3 in humans versus mice?",
        "target_paper": "A ganglioside-based immune checkpoint enables senescent cell.pdf",
        "expected_topics": ["Siglec-7 in human", "Siglec-E in mouse", "ITIM signaling"],
    },
    {
        "id": "Q8",
        "question": "What happens to NK cell degranulation and killing of senescent cells when ST8SIA1 is knocked down, or when cells are treated with neuraminidase or anti-GD3 antibodies?",
        "target_paper": "A ganglioside-based immune checkpoint enables senescent cell.pdf",
        "expected_topics": ["restored NK degranulation", "increased CD107a", "loss of immune evasion", "restored cytotoxicity"],
    },
    {
        "id": "Q9",
        "question": "What is the concept of 'senescent repair memory' in chronic disease, and how does failed resolution cause persistent organ dysfunction?",
        "target_paper": "Senescent repair memory in chronic d SO b Ageing Res Rev b 2.pdf",
        "expected_topics": ["repair memory", "failed resolution of inflammation/injury", "accumulation of senescent cells", "SASP perpetuation"],
    },
    {
        "id": "Q10",
        "question": "Based on the concept of senescent repair memory, how can clearing senescent cells or targeting their immune evasion break chronic dysfunction and allow tissue regeneration?",
        "target_paper": "Senescent repair memory in chronic d SO b Ageing Res Rev b 2.pdf",
        "expected_topics": ["senolytics / senoclearance", "restoring resolution", "breaking pathological memory", "regenerative restoration"],
    },
]


def build_settings() -> Settings:
    return Settings(
        llm="openai/deepseek-v4-flash-0731",
        summary_llm="openai/deepseek-v4-flash-0731",
        embedding="openai/text-embedding-v3",
        embedding_config={"batch_size": 8},
        parsing=ParsingSettings(
            multimodal=MultimodalOptions.OFF,
            use_doc_details=False,
        ),
    )


async def run_benchmark(output_json: Path):
    settings = build_settings()
    docs = Docs()

    paper_dir = Path("paper")
    target_pdfs = [
        "The ganglioside GD3 and its synthase (ST8SIA1) as novel sene.pdf",
        "A ganglioside-based immune checkpoint enables senescent cell.pdf",
        "Senescent repair memory in chronic d SO b Ageing Res Rev b 2.pdf",
    ]

    print("=" * 60)
    print("STEP 1: Indexing 3 key Osteoarthritis & Senescence Papers")
    print("=" * 60)
    start_index = time.time()
    for pdf_name in target_pdfs:
        pdf_path = paper_dir / pdf_name
        if not pdf_path.exists():
            print(f"Warning: {pdf_path} does not exist, skipping.")
            continue
        print(f"Indexing: {pdf_name}...")
        t0 = time.time()
        await docs.aadd(str(pdf_path), settings=settings)
        print(f"  Done in {time.time() - t0:.1f}s")

    print(f"All target documents indexed in {time.time() - start_index:.1f}s. Total docs: {len(docs.docs)}")
    print()

    print("=" * 60)
    print("STEP 2: Executing 10-Question Benchmark with PaperQA2")
    print("=" * 60)

    results = []
    for item in BENCHMARK_QUESTIONS:
        qid = item["id"]
        qtext = item["question"]
        print(f"\n--- Running [{qid}]: {qtext} ---")
        t0 = time.time()
        try:
            session = await docs.aquery(qtext, settings=settings)
            elapsed = time.time() - t0
            answer = session.formatted_answer
            raw_contexts = getattr(session, "contexts", [])
            contexts = []
            for c in raw_contexts:
                text_obj = getattr(c, "text", None)
                raw_text = (
                    getattr(text_obj, "text", str(text_obj))
                    if text_obj is not None
                    else getattr(c, "context", "")
                )
                source = (
                    getattr(text_obj, "name", "")
                    if text_obj is not None and hasattr(text_obj, "name")
                    else getattr(c, "context", str(c))
                )
                contexts.append({"source": source, "text": raw_text[:400]})

            has_citations = len(raw_contexts) > 0 and "pages" in answer

            res_entry = {
                "id": qid,
                "question": qtext,
                "target_paper": item["target_paper"],
                "expected_topics": item["expected_topics"],
                "answer": answer,
                "elapsed_seconds": round(elapsed, 2),
                "num_contexts": len(contexts),
                "has_citations": has_citations,
                "contexts": contexts[:3],
                "status": "SUCCESS",
            }
            print(f"Answer generated ({elapsed:.1f}s, {len(contexts)} contexts cited):")
            print(answer[:250].replace("\n", " ") + "...")
        except Exception as e:
            traceback.print_exc()
            print(f"ERROR on {qid}: {e}")
            res_entry = {
                "id": qid,
                "question": qtext,
                "target_paper": item["target_paper"],
                "expected_topics": item["expected_topics"],
                "answer": f"ERROR: {e}",
                "elapsed_seconds": 0,
                "num_contexts": 0,
                "has_citations": False,
                "contexts": [],
                "status": f"FAILED: {e}",
            }
        results.append(res_entry)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBenchmark completed. Results saved to: {output_json}")

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    cited_count = sum(1 for r in results if r["has_citations"])
    print(f"Total Questions: {len(results)}")
    print(f"Successful: {success_count}/{len(results)}")
    print(f"With Page-Grounded Citations: {cited_count}/{len(results)}")


if __name__ == "__main__":
    out_file = Path("data/benchmarks/paperqa2_10q_results.json")
    if len(sys.argv) > 1:
        out_file = Path(sys.argv[1])
    asyncio.run(run_benchmark(out_file))
