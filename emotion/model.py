"""Usage:
$ # If `emotion_local_files` is not in the same directory:
$ export EMOTION_LOCAL_FILES=/path/to/provided/emotion/local/files
$ python
Python 3.9.10 (main, Jan 15 2022, 11:48:04) 
[Clang 13.0.0 (clang-1300.0.29.3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from model import annotate
>>> annotate([{"id": 0, "contentText": "Emmanuel Macron, je t'adore!"}, {"id": 2, "contentText": "Emmanuel Macron, je te deteste!"}])
[{'id': 0, 'type': 'emotion-4.1', 'text': 'Anger, Hate, Contempt, Disgust', 'confidence': 0.012048926204442978, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 0, 'type': 'emotion-4.2', 'text': 'Embarrassment, Guilt, Shame, Sadness', 'confidence': 0.002244535367935896, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 0, 'type': 'emotion-4.3', 'text': 'Admiration, Love', 'confidence': 0.7399597764015198, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 0, 'type': 'emotion-4.4', 'text': 'Optimism, Hope', 'confidence': 0.0033456976525485516, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 0, 'type': 'emotion-4.5', 'text': 'Joy, Happiness', 'confidence': 0.04655738174915314, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 0, 'type': 'emotion-4.6', 'text': 'Pride, including National Pride', 'confidence': 0.011699127964675426, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 0, 'type': 'emotion-4.7', 'text': 'Fear, Pessimism', 'confidence': 0.002109723864123225, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 0, 'type': 'emotion-4.8', 'text': 'Amusement', 'confidence': 0.06591379642486572, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 0, 'type': 'emotion-4.9', 'text': 'Positive-other', 'confidence': 0.21505838632583618, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 0, 'type': 'emotion-4.10', 'text': 'Negative-other', 'confidence': 0.0053654685616493225, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 2, 'type': 'emotion-4.1', 'text': 'Anger, Hate, Contempt, Disgust', 'confidence': 0.8458896279335022, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 2, 'type': 'emotion-4.2', 'text': 'Embarrassment, Guilt, Shame, Sadness', 'confidence': 0.05854514613747597, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 2, 'type': 'emotion-4.3', 'text': 'Admiration, Love', 'confidence': 0.0027767797000706196, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 2, 'type': 'emotion-4.4', 'text': 'Optimism, Hope', 'confidence': 0.0019376322161406279, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 2, 'type': 'emotion-4.5', 'text': 'Joy, Happiness', 'confidence': 0.0019397599389776587, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 2, 'type': 'emotion-4.6', 'text': 'Pride, including National Pride', 'confidence': 0.0020261970348656178, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 2, 'type': 'emotion-4.7', 'text': 'Fear, Pessimism', 'confidence': 0.008506342768669128, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 2, 'type': 'emotion-4.8', 'text': 'Amusement', 'confidence': 0.006698955781757832, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 2, 'type': 'emotion-4.9', 'text': 'Positive-other', 'confidence': 0.002225079108029604, 'offset': [], 'providerName': 'ta1-usc-isi'},
{'id': 2, 'type': 'emotion-4.10', 'text': 'Negative-other', 'confidence': 0.2357044368982315, 'offset': [], 'providerName': 'ta1-usc-isi'}]
"""

import os
import sys
import warnings
from typing import List, Tuple, Dict, Any, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, XLMRobertaModel, PretrainedConfig
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
import logging


######################################################################
##### CONSTANTS
######################################################################

LOCAL_FILES_DIR = os.getenv("EMOTION_LOCAL_FILES")
if LOCAL_FILES_DIR is None:
    LOCAL_FILES_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "emotion_local_files"
    )

os.environ["EKPHRASIS_DIR"] = os.path.join(LOCAL_FILES_DIR, "ekphrasis")
MODEL_PATH = os.path.join(
    LOCAL_FILES_DIR, "xlm-roberta-twitter-goemotions-semeval-protagonist.pt"
)
TOKENIZER_PATH = os.path.join(LOCAL_FILES_DIR, "xlm-roberta-tokenizer")
CONFIG_PATH = os.path.join(LOCAL_FILES_DIR, "xlm-roberta-config")
MAX_LENGTH = 128


######################################################################
##### CONSTANTS end
######################################################################


######################################################################
##### DATA
######################################################################

# Disable
def block_print():
    sys.stdout = open(os.devnull, "w")


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    block_print()
    TOKENIZER = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH, local_files_only=True
    )
    PREPROCESSOR = TextPreProcessor(
        normalize=["url", "email", "phone", "user"],
        annotate={
            "hashtag",
            "elongated",
            "allcaps",
            "repeated",
            "emphasis",
            "censored",
        },
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
    ).pre_process_doc
    enable_print()


######################################################################
##### DATA end
######################################################################


######################################################################
##### PROMPT
######################################################################


def make_segment_labels(
    emo_list: List[str], tokenizer: "Tokenizer", label_list: List[List[str]]
) -> Tuple[str, List[List[str]]]:
    """Creates prompt and calculates the (sub)tokens in
    the prompt for each class for SpanEmo.

    Args:
        emo_list: the list of emotions/labels/classes.
        tokenizer: the tokenizer that will be used with the model.
        label_list: list of emotions per class.

    Returns:
        A tuple containing the prompt and the subtokens in the
        prompt for each class.
    """

    new_label_list = []
    for words in label_list:
        class_labels = []
        for word in words:
            class_labels.extend(tokenizer.tokenize(word))
        new_label_list.append(class_labels)
    label_list = new_label_list

    prompt = (
        " ".join(emo_list[:-1])
        + (" or " + emo_list[-1] if len(emo_list) > 1 else "")
        + "?"
    )

    return prompt, label_list


SCHEMA_FIELDS = [
    ("emotion-4.1", "Anger, Hate, Contempt, Disgust"),
    ("emotion-4.2", "Embarrassment, Guilt, Shame, Sadness"),
    ("emotion-4.3", "Admiration, Love"),
    ("emotion-4.4", "Optimism, Hope"),
    ("emotion-4.5", "Joy, Happiness"),
    ("emotion-4.6", "Pride, including National Pride"),
    ("emotion-4.7", "Fear, Pessimism"),
    ("emotion-4.8", "Amusement"),
    ("emotion-4.9", "Positive-other"),
    ("emotion-4.10", "Negative-other"),
]

EMOTION_CLUSTERS = [
    [
        "anger",
        "hate",
        "contempt",
        "disgust",
    ],  # hate, contempt not explicitly in semeval
    [
        "embarrassment",
        "guilt",
        "shame",
        "sadness",
    ],  # embarrassment, shame, guilt not explicitly in semeval
    [
        "admiration",
        "love",
    ],  # admiration not explicitly in semeval but exists in "trust"
    ["optimism", "hope"],  # hope not explicitly in semeval
    ["joy", "happiness"],  # happiness not explicitly in semeval
    ["pride"],  # not in semeval
    ["fear", "pessimism"],
    ["sarcasm", "amusement"],  # not in semeval
    ["positive"],  # not in semeval
    ["negative"],  # not in semeval
]

ELECTIONS_SEGMENT_MULTILINGUAL = [
    emotion for cluster in EMOTION_CLUSTERS for emotion in cluster
]

PROMPT, LABEL_NAMES = make_segment_labels(
    ELECTIONS_SEGMENT_MULTILINGUAL, TOKENIZER, EMOTION_CLUSTERS
)


######################################################################
##### MODEL
######################################################################


class BertEncoder(nn.Module):
    """Wrapper around a Bert-based model

    Attributes:
        bert: a Bert-based model, here `XLMRobertaModel`.
        feature_size: Bert's hidden feature size.
    """

    def __init__(self, config: "XLMRobertaConfig"):
        """Init.

        Args:
            config: config for XLMRoberta (consistent with
                any model that is going to be loaded afterwards).
        """
        super(BertEncoder, self).__init__()
        self.bert = XLMRobertaModel(config)

        self.feature_size = self.bert.config.hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            input_ids: list[str], list of tokenised sentences

        Returns:
            Last hidden representation of shape
            (batch_size, seq_length, hidden_dim)
        """
        last_hidden_state = self.bert(input_ids=input_ids).last_hidden_state
        return last_hidden_state


def _zero_check(fun):
    """Checks if the two first arguments of `fun`, which
    are expected to be tensors, have any values."""

    def wrapper(*args, **kwargs):
        if args[0].nelement() * args[1].nelement() == 0:
            return torch.tensor(0.0, device=args[0].device)
        return fun(*args, **kwargs)

    return wrapper


class SpanEmo(nn.Module):
    """SpanEmo.

    Attributes:
        bert: BertEncoder.
        ffn: final linear layers.
        All other attributes not used in this script.
    """

    def __init__(
        self,
        config: "XLMRobertaConfig",
        output_dropout=0.1,
        use_bce=True,
        corr_loss="exp_diff_loss",
        corr_loss_args=None,
        alpha=0.2,
    ):
        """Init.

        Args:
            config: config for XLMRoberta (consistent with
                any model that is going to be loaded afterwards).
        """
        super(SpanEmo, self).__init__()
        self.bert = BertEncoder(config)
        self.use_bce = use_bce
        self.corr_loss_type = corr_loss
        if not isinstance(corr_loss_args, list):
            corr_loss_args = [corr_loss_args]
        self.corr_loss_args = corr_loss_args
        self.alpha = alpha

        self.ffn = nn.Sequential(
            nn.Linear(self.bert.feature_size, self.bert.feature_size),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(self.bert.feature_size, 1),
        )

    def resize_token_embeddings(self, tokenizer):
        """Not used in this script"""
        self.bert.bert.resize_token_embeddings(len(tokenizer))

    def forward(self, batch, device, enable_corr_loss=True):
        """Not used in this script"""
        # prepare inputs and targets
        inputs, targets, lengths, label_idxs = batch
        inputs, num_rows = inputs.to(device), inputs.size(0)
        label_idxs = [inds.long().to(device) for inds in label_idxs]
        targets = targets.float().round().to(device)

        # Bert encoder
        # select span of labels to compare them with ground truth ones
        last_hidden_state = self.bert(inputs).index_select(
            dim=1, index=torch.cat(label_idxs)
        )

        cum_len = 0
        class_hidden = []
        for lbl_i in label_idxs:
            class_hidden.append(
                last_hidden_state.index_select(
                    dim=1,
                    index=torch.arange(
                        cum_len, cum_len + len(lbl_i), device=device
                    ),
                ).mean(dim=1)
            )
            cum_len += len(lbl_i)

        assert cum_len == len(torch.cat(label_idxs))

        last_hidden_state = torch.stack(class_hidden, dim=1)

        # FFN---> 2 linear layers---> linear layer + tanh---> linear layer
        logits = self.ffn(last_hidden_state).squeeze(-1)

        # Loss Function
        if self.use_bce or not enable_corr_loss:
            loss = F.binary_cross_entropy_with_logits(logits, targets).to(
                device
            )

        if self.corr_loss_type and enable_corr_loss:
            cl = self.corr_loss(
                logits,
                targets,
                fun=lambda a, b: self.__getattribute__(self.corr_loss_type)(
                    a, b, *self.corr_loss_args
                ),
            )
            if self.use_bce:
                loss = ((1 - self.alpha) * loss) + (self.alpha * cl)
            else:
                loss = cl

        y_pred = self.compute_pred(logits)
        return loss, num_rows, y_pred, targets.cpu().numpy()

    @staticmethod
    @_zero_check
    def exp_diff_loss(y_p, y_q, *args):
        return (y_p - y_q[..., None]).exp().mean()

    @staticmethod
    @_zero_check
    def mult_loss(y_p, y_q, pow=1):
        return (y_p**pow * (1 - y_q[..., None]) ** pow).mean()

    @staticmethod
    def corr_loss(
        y_hat,
        y_true,
        fun,
        reduction="mean",
    ):
        """
        Args:
            y_hat" model predictions, shape(batch, classes)
            y_true: target labels (batch, classes)
            reduction: whether to avg or sum loss
            fun: per sample loss given the predictions of
                negative and positive labels
        Returns:
            Loss.
        """
        loss_terms = torch.stack(
            [
                fun(
                    y_hat_i[y_true_i == 0].sigmoid(),
                    y_hat_i[y_true_i == 1].sigmoid(),
                )
                for y_hat_i, y_true_i in zip(y_hat, y_true)
            ]
        )
        loss = loss_terms.__getattribute__(reduction)()
        return loss

    @staticmethod
    def compute_pred(logits, threshold=0.5):
        """
        Args:
            logits: model predictions
            threshold: threshold value

        Returns:
            Predictions.
        """
        y_pred = torch.sigmoid(logits) > threshold
        return y_pred.float().cpu().numpy()


CONFIG = PretrainedConfig.from_pretrained(CONFIG_PATH, local_files_only=True)
MODEL = SpanEmo(CONFIG)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
MODEL.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = MODEL.to(DEVICE)
CONFIDENCE_SCORE = nn.Sigmoid()

logging.info("emotion model loaded...")

######################################################################
###### MODEL end
######################################################################


def format_output(ids: List[str], scores: List[float]) -> List[Dict[str, Any]]:
    """Formats the output scores.

    Args:
        ids: IDs of the tweet.
        scores: tensor of scores per tweet and emotion.

    Returns:
        A list with one dict per tweet of the form:
        {
            "id": "swqeivfbo23ierg74befiod",
            "type": "emotion-4.7",
            "text": "Fear, Pessimism",
            "confidence": 0.34553456,
            "providerName": "ta1-usc-isi",
        }
    """
    outputs = []
    for _id, tweet_scores in zip(ids, scores):
        for (_type, text), score in zip(SCHEMA_FIELDS, tweet_scores):
            outputs.append(
                {
                    "id": _id,
                    "type": _type,
                    "text": text,
                    "confidence": score,
                    "providerName": "ta1-usc-isi",
                }
            )

    return outputs


def annotate(
    tweets: Union[Dict[str, Any], List[Dict[str, Any]]],
    batch_size: Optional[int] = 256,
) -> List[Dict[str, Any]]:
    """Wrapper around model, tokenizer and preprocessing
    to compute emotion clusters in a tweet.

    Args:
        tweets: List of dicts, one per tweet, or just one
            dict for one tweet.
        batch_size: evaluation batch size, default set
            for less than 5GB RAM usage.

    Returns:
        A list with one dict per tweet of the form:
        {
            "id": "swqeivfbo23ierg74befiod",
            "type": "emotion-4.7",
            "text": "Fear, Pessimism",
            "confidence": 0.34553456,
            "providerName": "ta1-usc-isi",
        }
    """

    if isinstance(tweets, dict):
        tweets = [tweets]

    ids = [tweet["id"] for tweet in tweets]
    text = [tweet["contentText"] for tweet in tweets]
    text = [
        TOKENIZER.encode_plus(
            PROMPT,
            " ".join(PREPROCESSOR(tweet)),
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
        )
        for tweet in text
    ]

    example_inds = []
    label_tokens = TOKENIZER.convert_ids_to_tokens(text[0]["input_ids"])

    # if tokenizer breaks word into multiple subtokens, grab all
    for labels in LABEL_NAMES:
        label_inds = []
        for label in labels:
            label_idx = label_tokens.index(label)
            # pop w/o changing indices
            # so that same label names map to their actual index
            label_tokens[label_idx] = None
            label_inds.append(label_idx)
        example_inds.append(torch.tensor(label_inds, dtype=torch.long))
    index = torch.cat(example_inds).to(DEVICE)

    input_ids = torch.stack(
        [torch.tensor(tweet["input_ids"], dtype=torch.long) for tweet in text]
    )

    data_loader = DataLoader(TensorDataset(input_ids), batch_size=batch_size)

    scores = []

    for inp in data_loader:
        inp = inp[0].to(DEVICE)  # TensorDataset always returns tuple

        with torch.no_grad():
            last_hidden_state = MODEL.bert(inp).index_select(dim=1, index=index)

        # aggregate contextual embeddings on the cluster level
        # using the computed indices per cluster
        cum_len = 0
        class_hidden = []
        for lbl_i in example_inds:

            cls_index = torch.arange(cum_len, cum_len + len(lbl_i)).to(DEVICE)
            class_hidden.append(
                last_hidden_state.index_select(dim=1, index=cls_index).mean(
                    dim=1
                )
            )
            cum_len += len(lbl_i)

        last_hidden_state = torch.stack(class_hidden, dim=1)

        with torch.no_grad():
            scores.extend(
                CONFIDENCE_SCORE(MODEL.ffn(last_hidden_state).squeeze(-1))
                .to("cpu")
                .tolist()
            )

    return format_output(ids, scores)