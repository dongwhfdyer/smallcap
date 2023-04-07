from pathlib import Path

from transformers import AutoModelForCausalLM, PreTrainedModel, AutoConfig, AutoModel

# model = PreTrainedModel.from_pretrained("experiments/rag_7M_gpt2/checkpoint-22140")
from src.vision_encoder_decoder import SmallCapConfig

checkpoint_path = Path("experiments/rag_7M_gpt2/checkpoint-19926")

# checkpoint_path = Path("experiments/rag_7M_gpt2/checkpoint-22140")
def register_model_and_config():
    from transformers import AutoModelForCausalLM
    from src.vision_encoder_decoder import SmallCap, SmallCapConfig
    from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel

    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)
register_model_and_config()
config = AutoConfig.from_pretrained(checkpoint_path / 'config.json')
model = AutoModel.from_pretrained(checkpoint_path)
model.config = config
model.eval()
# from src.vision_encoder_decoder import SmallCap

# smallcap = SmallCap(checkpoint_path / 'config.json')

print("--------------------------------------------------")