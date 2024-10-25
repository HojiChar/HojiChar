import hashlib
import logging
import os
import time
from os import PathLike
from pathlib import Path
from typing import Any, Tuple, Union

try:
    import requests
    from fasttext import load_model  # type: ignore

    is_loaded_extras = True
except ImportError:
    is_loaded_extras = False

from tqdm import tqdm

from hojichar import Document, Filter

logger = logging.getLogger(__name__)


FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_CHECKSUM = "01810bc59c6a3d2b79c79e6336612f65"


def _download_with_progress_bar(
    download_url: str, save_path: Union[str, PathLike], retries: int = 3, delay: float = 1.0
) -> None:
    # HACK type hint `os.PathLike[str]` is not allowed in Python 3.8 or older.
    # So I write Union[str, PathLike]. In the future, I will use `os.PathLike[str]` or simply Path.
    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
    except requests.RequestException as e:
        if retries > 0:
            logger.warning(
                f"Download failed, retrying in {delay} seconds... ({retries} retries left)"
            )
            time.sleep(delay)
            _download_with_progress_bar(download_url, save_path, retries - 1, delay)
        else:
            logger.error(f"Download failed after retries: {e}")
            raise


def _get_md5_hash_of_file(file_path: Union[str, PathLike]) -> str:
    """
    Function to calculate the MD5 hash of a file.

    Read the file in chunks to avoid loading large files into memory.
    cf. https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def _download_fasttext_model(model_path: Union[str, PathLike]) -> None:
    logger.info(f"Downloading fasttext model from {FASTTEXT_MODEL_URL} to {model_path}...")
    _download_with_progress_bar(FASTTEXT_MODEL_URL, model_path)


class LanguageIdentificationByFastText(Filter):
    """
    A filter that removes non-Japanese text
    using Language IDentification (LID).
    Download the fastText model for LID: https://fasttext.cc/docs/en/language-identification.html
    The available languages are:
        af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce ceb
        ckb co cs cv cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom
        gu gv he hi hif hr hsb ht hu hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv
        kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn
        nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah
        sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec
        vep vi vls vo wa war wuu xal xmf yi yo yue zh
    """

    def __init__(
        self,
        language: str,
        lang_score_threshold: float = 0.50,
        model_path: Union[str, PathLike, None] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            language: the language to be accepted.
            lang_score_threshold: the document whose fasttext score is
              below this threshold will be discarded.
              A default value is 0.50, which is empirically very generous value,
              i.e., almost all documents are accepted if the document
              is classified as japanese text.
            model_path: The directory path, which the model saved in.
                If None, the model will be saved in the current directory.
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        if not is_loaded_extras:
            raise ImportError(
                "The `fasttext` package is required to use this filter. "
                "Please install it by running `pip install hojichar[all]`"
                "or `pip install fasttext requests`."
            )

        self.lang_score_threshold = lang_score_threshold
        self.language = language

        self.model_path = Path(model_path) if model_path else Path(os.getcwd()) / "lid.176.bin"

        if not self.model_path.exists():
            logger.info("Fasttext model file was not found.")
            _download_fasttext_model(self.model_path)

        assert _get_md5_hash_of_file(self.model_path) == MODEL_CHECKSUM, (
            f"Checksum of the downloaded model file does not match the expected value. "
            f"Expected: {MODEL_CHECKSUM}, "
            f"Actual: {_get_md5_hash_of_file(self.model_path)}"
        )
        self.model = load_model(str(self.model_path))

    def _predict_language(self, text: str) -> Tuple[str, float]:
        # fasttext cannot handle multiline input
        # so we must remove the newline character
        text = text.strip().replace("\n", " ")
        pred = self.model.predict(text)
        pred_lang = pred[0][0].replace("__label__", "")
        pred_score = pred[1][0]
        return pred_lang, pred_score

    def apply(self, doc: Document) -> Document:
        pred_lang, score = self._predict_language(doc.text)
        if not (pred_lang == self.language and score >= self.lang_score_threshold):
            doc.is_rejected = True
        return doc


class AcceptJapaneseByFastText(LanguageIdentificationByFastText):
    """
    A filter that removes non-Japanese text via Language Identification (LID) by FastText.

    >>> AcceptJapaneseByFastText().apply(Document("This is English document")).is_rejected
    True
    >>> AcceptJapaneseByFastText().apply(Document("自然言語処理大好き！")).is_rejected
    False
    >>> AcceptJapaneseByFastText().apply(Document("快三手机投注平台代理")).is_rejected
    True
    """

    def __init__(
        self,
        lang_score_threshold: float = 0.50,
        model_path: Union[str, PathLike, None] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__("ja", lang_score_threshold, model_path, *args, **kwargs)
