from llmsearch.config import Config, get_config

DEFAULT_TEMPLATE_FN = "sample_templates/config_template.yaml"

def test_config():
    config = get_config(DEFAULT_TEMPLATE_FN)