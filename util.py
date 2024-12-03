import LLM

def create_model(base: str, id: str, lora: str = None) -> LLM.LLM:
    package = __import__("LLM")
    model_class = getattr(package, base)
    return model_class(id, lora)