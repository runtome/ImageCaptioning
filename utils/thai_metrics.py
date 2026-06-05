from typing import Dict, List


def thai_tokenize(text: str) -> List[str]:
    """Word-segment Thai text using PyThaiNLP's newmm engine."""
    from pythainlp.tokenize import word_tokenize
    tokens = word_tokenize(text.strip(), engine="newmm", keep_whitespace=False)
    return [t for t in tokens if t.strip()]


def compute_metrics(
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """
    Compute BLEU-4, CIDEr, and METEOR for Thai captions.

    Args:
        predictions: generated captions, one per image
        references:  list of reference caption lists (3 per image)
    """
    results: Dict[str, float] = {}

    tok_hyps = [thai_tokenize(p) for p in predictions]
    tok_refs = [[thai_tokenize(r) for r in refs] for refs in references]

    # BLEU-4
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        bleu4 = corpus_bleu(
            tok_refs,
            tok_hyps,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )
        results["bleu4"] = round(bleu4 * 100, 2)
    except Exception as e:
        results["bleu4"] = -1.0
        print(f"[metrics] BLEU-4 failed: {e}")

    # METEOR — NLTK expects token lists, not joined strings
    try:
        from nltk.translate.meteor_score import meteor_score as _meteor
        scores = []
        for hyp_tokens, ref_token_lists in zip(tok_hyps, tok_refs):
            score = _meteor(ref_token_lists, hyp_tokens)
            scores.append(score)
        results["meteor"] = round(sum(scores) / len(scores) * 100, 2) if scores else -1.0
    except Exception as e:
        results["meteor"] = -1.0
        print(f"[metrics] METEOR failed: {e}")

    # CIDEr (requires pycocoevalcap)
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {
            i: [" ".join(thai_tokenize(r)) for r in refs]
            for i, refs in enumerate(references)
        }
        res = {
            i: [" ".join(thai_tokenize(p))]
            for i, p in enumerate(predictions)
        }
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        results["cider"] = round(score * 100, 2)
    except ImportError:
        results["cider"] = -1.0
        print("[metrics] CIDEr skipped — install pycocoevalcap")
    except Exception as e:
        results["cider"] = -1.0
        print(f"[metrics] CIDEr failed: {e}")

    return results
