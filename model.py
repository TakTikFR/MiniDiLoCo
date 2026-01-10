import os
from transformers import AutoModelForCausalLM, AutoConfig

def get_gpt2_model():
    """ Return the pretrained gpt2 model. """
    
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path='gpt2')
    RANK = int(os.environ['LOCAL_RANK'])
    model = AutoModelForCausalLM.from_config(config).cuda(RANK)

    return model