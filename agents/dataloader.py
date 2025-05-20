from datasets import load_dataset
from typing import Dict, List, Callable


def get_dataset_loader() -> Dict[str, Callable]:
    """
    Returns a dictionary of dataset loaders by name.
    """
    return {
        "rotowire": lambda: load_dataset("mrm8488/rotowire-sbnation", trust_remote_code=True),
        "turku_hockey": lambda: load_dataset("GEM/turku_hockey_data2text", trust_remote_code=True),
        "totto": lambda: load_dataset("GEM/totto", trust_remote_code=True),
        "sportsett_basketball": lambda: load_dataset("GEM/sportsett_basketball", trust_remote_code=True),
        "webnlg": lambda: load_dataset("GEM/web_nlg", "en", trust_remote_code=True),
        "conversational_weather": lambda: load_dataset("GEM/conversational_weather", trust_remote_code=True),
        "dart": lambda: load_dataset("GEM/dart", trust_remote_code=True),
        "mlb": lambda: load_dataset("GEM/mlb_data_to_text", trust_remote_code=True),
    }


def load_dataset_by_name(name: str):
    """
    Loads a specific dataset by name.
    """
    loaders = get_dataset_loader()
    if name not in loaders:
        raise ValueError(f"Dataset '{name}' is not supported. Choose from: {list(loaders.keys())}")
    print(f"Loading dataset: {name}")
    return loaders[name]()  # returns a DatasetDict


def extract_example(dataset_name: str, example: Dict) -> Dict:
    """
    Normalize one data example to a common structure: {input, target/reference}
    """
    if dataset_name == "rotowire":
        return {
            "input": example.get("box_score", ""),
            "target": ' '.join(example.get("summary", []))
        }
    elif dataset_name in ["totto", "sportsett_basketball", "mlb"]:
        return {
            "input": example.get("linearized_input", ""),
            "target": example.get("target", ""),
            "references": example.get("references", [])
        }
    elif dataset_name == "webnlg":
        return {
            "input": example.get("input", ""),
            "target": example.get("target", "")
        }
    elif dataset_name == "conversational_weather":
        return {
            "input": example.get("tree_str_mr", example.get("user_query", "")),
            "target": example.get("target", ""),
            "references": example.get("references", [])
        }
    elif dataset_name == "dart":
        return {
            "input": str(example.get("tripleset", "")),
            "target": example.get("target", ""),
            "references": example.get("references", [])
        }
    elif dataset_name == "turku_hockey":
        return {
            "input": example.get("input", ""),
            "target": example.get("target", "")
        }
    else:
        return {
            "input": example.get("input", ""),
            "target": example.get("target", "")
        }


# Example usage
if __name__ == "__main__":
    name = "dart"  # Change this to the dataset you want to load
    data = load_dataset_by_name(name)
    sample = extract_example(name, data["test"][0])
    print(sample)
