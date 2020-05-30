from transformers import (
    AutoModel,
    AutoConfig,
)

pretrained = 'bert-base-uncased'

model_from_pretrained = AutoModel.from_pretrained(pretrained)
model_from_config = AutoModel.from_config(AutoConfig.from_pretrained(pretrained))

model_from_pretrained_params = list(model_from_pretrained.parameters())
model_from_config_params = list(model_from_config.parameters())

assert len(model_from_pretrained_params) == len(model_from_config_params)

model_from_pretrained_first_param = model_from_pretrained_params[0][0][0]
model_from_config_first_param = model_from_config_params[0][0][0]

assert model_from_pretrained_first_param == model_from_config_first_param, (
    f'{model_from_pretrained_first_param} != {model_from_config_first_param}'
)
