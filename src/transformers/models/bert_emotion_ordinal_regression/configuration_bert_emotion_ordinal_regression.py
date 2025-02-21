from ..bert import BertConfig
from ...configuration_utils import PretrainedConfig


EMOTION_CATEGORIES_MAPPING = [
    ("Amusement", 0, 4),
    ("Elation", 0, 4),
    ("Pleasure/Ecstasy", 0, 4),
    ("Contentment", 0, 4),
    ("Thankfulness/Gratitude", 0, 4),
    ("Affection", 0, 4),
    ("Infatuation", 0, 4),
    ("Hope/Enthusiasm/Optimism", 0, 4),
    ("Triumph", 0, 4),
    ("Pride", 0, 4),
    ("Interest", 0, 4),
    ("Awe", 0, 4),
    ("Astonishment/Surprise", 0, 4),
    ("Concentration", 0, 4),
    ("Contemplation", 0, 4),
    ("Relief", 0, 4),
    ("Longing", 0, 4),
    ("Teasing", 0, 4),
    ("Impatience and Irritability", 0, 4),
    ("Sexual Lust", 0, 4),
    ("Doubt", 0, 4),
    ("Fear", 0, 4),
    ("Distress", 0, 4),
    ("Confusion", 0, 4),
    ("Embarrassment", 0, 4),
    ("Shame", 0, 4),
    ("Disappointment", 0, 4),
    ("Sadness", 0, 4),
    ("Bitterness", 0, 4),
    ("Contempt", 0, 4),
    ("Disgust", 0, 4),
    ("Anger", 0, 4),
    ("Malevolence/Malice", 0, 4),
    ("Sourness", 0, 4),
    ("Pain", 0, 4),
    ("Helplessness", 0, 4),
    ("Fatigue/Exhaustion", 0, 4),
    ("Emotional Numbness", 0, 4),
    ("Intoxication/Altered States of Consciousness", 0, 4),
    ("Jealousy & Envy", 0, 4),
]

# Mapping table for dimensions.
# Each tuple is (key, min_value, max_value)
EMOTION_DIMENSIONS_MAPPING = [
    ("Valence", -3.0, 3.0),  # Range: -3 to +3
    ("Arousal", 0.0, 4.0),  # Range: 0 to 4
    ("Submissive vs. Dominant", -3.0, 3.0),  # Range: -3 to +3
    ("Age", 0.0, 6.0),  # Range: 0 to 6
    ("Gender", -2.0, 2.0),  # Range: -2 to +2
    ("Serious vs. Humorous", 0.0, 4.0),  # Range: 0 to 4
    ("Vulnerable vs. Emotionally Detached", 0.0, 4.0),  # Range: 0 to 4
    ("Confident vs. Hesitant", 0.0, 4.0),  # Range: 0 to 4
    ("Warm vs. Cold", -2.0, 2.0),  # Range: -2 to +2
    ("Monotone vs. Expressive", 0.0, 4.0),  # Range: 0 to 4
    ("High-Pitched vs. Low-Pitched", 0.0, 4.0),  # Range: 0 to 4
    ("Soft vs. Harsh", -2.0, 2.0),  # Range: -2 to +2
    ("Authenticity", 0.0, 4.0),  # Range: 0 to 4
    ("Recording Quality", 0.0, 4.0),  # Range: 0 to 4
    ("Background Noise", 0.0, 3.0),  # Range: 0 to 3
]



class BertForMultiOutputOrdinalRegressionConfig(PretrainedConfig):
    
    model_type = "bert-emotion-ordinal-regression"
    def __init__(
        self,
        bert_config: PretrainedConfig = None,
        categories_mapping: list[tuple[str, float, float]] = EMOTION_CATEGORIES_MAPPING,
        dimensions_mapping: list[tuple[str, float, float]] = EMOTION_DIMENSIONS_MAPPING,
        hidden_dropout_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # If bert_config isn't provided, load the default from pretrained.
        if bert_config is None:
            bert_config = BertConfig.from_pretrained("bert-base-uncased")
        # If bert_config is a dict (from deserialization), convert it.
        elif isinstance(bert_config, dict):
            bert_config = BertConfig.from_dict(bert_config)
        self.categories_mapping = categories_mapping
        self.dimensions_mapping = dimensions_mapping
        self.bert_config = bert_config
        self.hidden_dropout_prob = hidden_dropout_prob


__all__ = [
    "BertForMultiOutputOrdinalRegressionConfig",
    "EMOTION_CATEGORIES_MAPPING",
    "EMOTION_DIMENSIONS_MAPPING",
]