from hojichar.filters.document_filters import MaskPersonalInformation


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
