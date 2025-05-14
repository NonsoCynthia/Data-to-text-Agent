import nltk
# nltk.download('all')
import torch
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
import sacrebleu
from comet import download_model, load_from_checkpoint
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import evaluate

# # Download NLTK resources
# nltk.download('punkt', quiet=True)

# Load once: COMET model
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)
comet_model.eval()

# Load once: BLEURT model
bleurt_config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12')
bleurt_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')
bleurt_model.eval()

# Load once: BERTScore evaluator
bertscore_metric = evaluate.load("bertscore")


def evaluate_single(ground_truth: str, prediction: str, source: str = "") -> dict:
    """
    Evaluate a single prediction against the ground truth using multiple metrics.

    Args:
        ground_truth (str): The reference text.
        prediction (str): The predicted/generated text.
        source (str): The source text (needed for COMET).

    Returns:
        dict: Dictionary containing evaluation scores.
    """
    # BLEU
    bleu = sacrebleu.raw_corpus_bleu([prediction], [[ground_truth]], 0.01).score

    # METEOR
    gt_tokens = word_tokenize(ground_truth)
    pred_tokens = word_tokenize(prediction)
    meteor = single_meteor_score(gt_tokens, pred_tokens)

    # ROUGE
    rouge_s = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = rouge_s.score(ground_truth, prediction)
    rouge_f1 = sum([
        rouge['rouge1'].fmeasure,
        rouge['rouge2'].fmeasure,
        rouge['rougeL'].fmeasure
    ]) / 3

    # COMET
    comet_input = [{"src": source or ground_truth, "mt": prediction, "ref": ground_truth}]
    comet_score = comet_model.predict(comet_input, gpus=0)[0][0]

    # BLEURT
    with torch.no_grad():
        bleurt_inputs = bleurt_tokenizer(ground_truth, prediction, return_tensors='pt', padding=True)
        bleurt_output = bleurt_model(**bleurt_inputs).logits.flatten().item()

    # BERTScore
    bert_out = bertscore_metric.compute(predictions=[prediction], references=[ground_truth], lang="en")
    bert_f1 = bert_out["f1"][0]

    return {
        "BLEU": bleu,
        "METEOR": meteor,
        "ROUGE-F1": rouge_f1,
        "COMET": comet_score,
        "BLEURT": bleurt_output,
        "BERTScore-F1": bert_f1
    }


if __name__ == "__main__":
    gt = "Gauff, just 15, shocks 5-time champ Venus, 39, at Wimbledon"
    pred = "15-year-old Gauff beats 5-time Wimbledon champ Venus"
    src = "Cori Gauff, 15, defeated Venus Williams at Wimbledon."

    scores = evaluate_single(gt, pred, src)
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
