import random
import re
import time

import pytest

from hojichar.filters.document_filters import MaskPersonalInformation

# The e-mail pattern before the local part was bounded to 64 characters.
LEGACY_EMAIL_PAT = re.compile(
    r"[a-zA-Z0-9!#$%&'*+\-/=?^_`{|}~.]+@[A-Za-z0-9!#$%&'*+\-/=?^_`{|}~.]+(\.[A-Za-z0-9\-]+)"
)


PHONE_PAT = MaskPersonalInformation().phone_pat


def legacy_mask(text: str) -> str:
    """The output of `MaskPersonalInformation` with the legacy e-mail pattern."""
    return LEGACY_EMAIL_PAT.sub(r"xxxx@yyy\1", PHONE_PAT.sub(r"\1XXXX", text))


NORMAL_EMAIL_TEXTS = [
    "hoge@example.com",
    "hoge.fuga.111@foo.com",
    "hogehoge@example.ne.jp",
    "*-+|#$.@!!.com",
    "first.last+tag@sub.example.co.uk",
    "Email john_doe-99@mail-server.example.org.",
    "連絡先: info@example.jp / support@example.com",
    "メールはtaro.yamada@example.co.jpまで",
    '<a href="mailto:taro@example.jp">taro@example.jp</a>',
    "Message-ID: <CAF+abc123.def456@mail.gmail.com>",
    "git clone git@github.com:HojiChar/HojiChar.git",
    "a@b.c@d.e and a@b.com/c@d.org",
    "user@localhost, @mention, foo@ and @bar.com",
    "x" * 64 + "@example.com",
    "宛先\nsales@example.com\ncc: hr@example.co.jp\n",
]


class TestMaskPhoneNumber:
    filt = MaskPersonalInformation()

    def test_japanese_landline_number(self):
        """
        Japanese landline numbers have following structur:
            0[Area code(市外局番)] - [City code(市内局番)] - [Subscriber number]
            The area code and city code have a total of 5 digits.
        The filter masking subscriber number.
        """
        # With hyphen
        assert self.filt("06-1234-5678") == "06-1234-XXXX"
        assert self.filt("075-123-4567") == "075-123-XXXX"
        assert self.filt("0166-12-3456") == "0166-12-XXXX"
        assert self.filt("09808-1-2345") == "09808-1-XXXX"
        # With none char
        assert self.filt("0612345678") == "061234XXXX"
        assert self.filt("0751234567") == "075123XXXX"
        assert self.filt("0166123456") == "016612XXXX"
        assert self.filt("0980812345") == "098081XXXX"
        # With blank char
        assert self.filt("06 1234 5678") == "06 1234 XXXX"
        assert self.filt("075 123 4567") == "075 123 XXXX"
        assert self.filt("0166 12 3456") == "0166 12 XXXX"
        assert self.filt("09808 1 2345") == "09808 1 XXXX"

    def test_japanese_mobile_phone_number(self):
        assert self.filt("080-1234-5678") == "080-1234-XXXX"
        assert self.filt("08012345678") == "0801234XXXX"
        assert self.filt("080 1234 5678") == "080 1234 XXXX"

    def test_japanese_phone_number_international_prefix(self):
        assert self.filt("+81-6-1234-5678") == "+81-6-1234-XXXX"
        assert self.filt("+81-75-1234-5678") == "+81-75-1234-XXXX"
        assert self.filt("+81-166-12-3456") == "+81-166-12-XXXX"
        assert self.filt("+81-9808-1-2345") == "+81-9808-1-XXXX"
        assert self.filt("+81-90-1234-5678") == "+81-90-1234-XXXX"

        assert self.filt("+81612345678") == "+8161234XXXX"
        assert self.filt("+817512345678") == "+81751234XXXX"
        assert self.filt("+81166123456") == "+8116612XXXX"
        assert self.filt("+810980812345") == "+81098081XXXX"
        assert self.filt("+819012345678") == "+81901234XXXX"

        assert self.filt("+81 6 1234 5678") == "+81 6 1234 XXXX"
        assert self.filt("+81 75 1234 5678") == "+81 75 1234 XXXX"
        assert self.filt("+81 166 12 3456") == "+81 166 12 XXXX"
        assert self.filt("+81 09808 1 2345") == "+81 09808 1 XXXX"
        assert self.filt("+81 90 1234 5678") == "+81 90 1234 XXXX"

    def test_embedded_in_text(self):
        assert (
            self.filt("Call 075-123-4567 if something is wrong")
            == "Call 075-123-XXXX if something is wrong"
        )
        assert (
            self.filt("なにかあったら08012345678まで電話してください.")
            == "なにかあったら0801234XXXXまで電話してください."
        )


class TestMaskEmailAddress:
    filt = MaskPersonalInformation()

    def test_email_address(self):
        assert self.filt("hoge@example.com") == "xxxx@yyy.com"
        assert self.filt("hoge.fuga.111@foo.com") == "xxxx@yyy.com"
        assert self.filt("hogehoge@example.ne.jp") == "xxxx@yyy.jp"
        assert self.filt("*-+|#$.@!!.com") == "xxxx@yyy.com"

    def test_embedded_in_text(self):
        assert (
            self.filt("何かあれば hogehoge@example.ne.jp まで連絡")
            == "何かあれば xxxx@yyy.jp まで連絡"
        )

    @pytest.mark.parametrize("text", NORMAL_EMAIL_TEXTS)
    def test_same_output_as_legacy_pattern(self, text):
        assert self.filt(text) == legacy_mask(text)

    def test_same_output_as_legacy_pattern_on_random_short_texts(self):
        # A local part cannot exceed 64 characters in these texts, so the outputs must agree.
        rng = random.Random(0)
        alphabet = "ab9.-_!+/@@.. あ\n"
        for _ in range(2000):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 64)))
            assert self.filt(text) == legacy_mask(text), text

    def test_long_local_part(self):
        # Only the 64 characters before "@" are taken as the local part.
        assert self.filt("a" * 70 + "@example.com") == "a" * 6 + "xxxx@yyy.com"

    def test_long_runs_around_at_sign_are_fast(self):
        # Long runs of address characters around "@" and no domain with a dot. The legacy
        # pattern rescanned each run from every position in it: about 10 seconds on a laptop,
        # against 0.04 seconds now.
        text = ("a" * 20_000 + "@" + "b" * 20_000 + " ") * 5
        start = time.perf_counter()
        assert self.filt(text) == text
        assert time.perf_counter() - start < 2.0
