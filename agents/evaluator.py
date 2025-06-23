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

def ensure_list(x):
    return x if isinstance(x, list) else [x]

def evaluate_single(references, prediction, sources=None) -> dict:
    """
    Evaluate a single prediction against multiple ground truth references.

    Args:
        references (List[str] or str): Reference(s) to compare against.
        prediction (str): Model prediction.
        sources (List[str] or str or None): Source text(s) for COMET.

    Returns:
        dict: Dictionary containing averaged evaluation scores.
    """
    references = ensure_list(references)
    sources = ensure_list(sources) if sources is not None else references

    # BLEU (supports multiple references)
    bleu = sacrebleu.raw_corpus_bleu([prediction], [[ref] for ref in references]).score/100
    # bleu = sentence_bleu([word_tokenize(ref) for ref in references], word_tokenize(prediction), smoothing_function=smoother.method1)

    # METEOR (only one reference supported)
    meteor = single_meteor_score(word_tokenize(references[0]), word_tokenize(prediction))


    # ROUGE (using only the first reference)
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = rouge_scorer_obj.score(references[0], prediction)
    rouge_f1 = sum([rouge['rouge1'].fmeasure, rouge['rouge2'].fmeasure, rouge['rougeL'].fmeasure]) / 3

    # COMET (evaluate using the first source and reference)
    comet_input = [{"src": sources[0], "mt": prediction, "ref": references[0]}]
    use_gpu = torch.cuda.is_available()
    comet_score = comet_model.predict(comet_input, gpus=1 if use_gpu else 0)[0][0]

    # BLEURT (evaluate using the first reference)
    with torch.no_grad():
        inputs = bleurt_tokenizer(
            references[0],
            prediction,
            return_tensors='pt',
            padding=True,
            truncation=True,  #truncate to max length
            max_length=512).to(bleurt_model.device)

        bleurt_output = bleurt_model(**inputs).logits.flatten().item()

    # BERTScore (supports multiple references)
    bert_out = bertscore_metric.compute(
        predictions=[prediction], references=[references], lang="en"
    )
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
    gt = ["Gauff, just 15, shocks 5-time champ Venus, 39, at Wimbledon"]
    pred = "15-year-old Gauff beats 5-time Wimbledon champ Venus"
    src = ["Cori Gauff, 15, defeated Venus Williams at Wimbledon."]

    scores = evaluate_single(gt, pred, src)
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
