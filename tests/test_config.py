from llmsearch.config import get_config

DEFAULT_TEMPLATE_FN = "sample_templates/generic/config_template.yaml"


def test_config():
    config = get_config(DEFAULT_TEMPLATE_FN)
