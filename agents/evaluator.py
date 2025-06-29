import nltk
# nltk.download('all')
import torch
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smoother = SmoothingFunction()
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
import sacrebleu
from comet import download_model, load_from_checkpoint
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
import evaluate

# # Download NLTK resources
# nltk.download('punkt', quiet=True)

def ensure_list(x):
    return x if isinstance(x, list) else [x]


class BatchEvaluator:
    """Load metric models once and reuse them for every example."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- Heavy checkpoints (loaded once) ----------
        comet_ckpt = download_model("Unbabel/wmt22-comet-da")
        self.comet = load_from_checkpoint(comet_ckpt).to(self.device)
        self.comet.eval()

        self.bleurt_tok = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20-D12")
        self.bleurt = (
            BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20-D12")
            .to(self.device)
        )
        self.bleurt.eval()

        self.bertscore = evaluate.load("bertscore")

    def score(self, references, prediction, sources=None) -> dict:
        """Return a dict with all metrics normalised to the 0‒1 interval."""
        references = ensure_list(references)
        sources = ensure_list(sources) if sources is not None else references

        # BLEU (supports multiple references)
        # bleu = sacrebleu.raw_corpus_bleu([prediction], [[ref] for ref in references]).score/100
        # bleu = sentence_bleu([word_tokenize(ref) for ref in references], word_tokenize(prediction), smoothing_function=smoother.method1)
        bleu = sacrebleu.sentence_bleu(
                                    prediction,           # hypothesis string
                                    references,           # list of reference variants
                                    smooth_method="exp",  # classic smoothing
                                    tokenize="intl",      # standard detok rules
                                    lowercase=False,
                                ).score / 100.0

        # # METEOR (single reference, already 0‑1)
        # meteor = single_meteor_score(
        #     word_tokenize(references[0]), word_tokenize(prediction)
        # )

        # # ROUGE (first reference, F‑measure already 0‑1)
        # rouge_scores = rouge_scorer.RougeScorer(
        #     ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        # ).score(references[0], prediction)
        # rouge_f1 = sum(r.fmeasure for r in rouge_scores.values()) / 3

        # # COMET
        # inp = [{"src": sources[0], "mt": prediction, "ref": references[0]}]
        # comet_score = self.comet.predict(inp, gpus=1 if self.device == "cuda" else 0)[0][0]

        # # BLEURT
        # with torch.no_grad():
        #     bleurt_inputs = self.bleurt_tok(
        #         references[0],
        #         prediction,
        #         return_tensors="pt",
        #         padding=True,
        #         truncation=True,
        #         max_length=512,
        #     ).to(self.device)
        #     bleurt_score = self.bleurt(**bleurt_inputs).logits.flatten().item()

        # # BERTScore
        # bert_f1 = self.bertscore.compute(
        #     predictions=[prediction], references=[references], lang="en"
        # )["f1"][0]

        return {
            "BLEU": bleu,
            # "METEOR": meteor,
            # "ROUGE-F1": rouge_f1,
            # "COMET": comet_score,
            # "BLEURT": bleurt_score,
            # "BERTScore-F1": bert_f1,
        }


def evaluate_single(references, prediction, sources=None):
    _default_evaluator = BatchEvaluator()
    """Legacy wrapper so existing code keeps working."""
    return _default_evaluator.score(references, prediction, sources)


if __name__ == "__main__":
    references = ["Gauff, just 15, shocks 5-time champ Venus, 39, at Wimbledon"]
    prediction  = "15-year-old Gauff beats 5-time Wimbledon champ Venus"
    source      = ["Cori Gauff, 15, defeated Venus Williams at Wimbledon."]

    evaluator = BatchEvaluator()
    scores = evaluator.score(references, prediction, source)

    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")
