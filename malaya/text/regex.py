# this file to store possible regex

_ltr_emoticon = [
    # optional hat
    r'(?:(?<![a-zA-Z])[DPO]|(?<!\d)[03]|[|}><=])?',
    # eyes
    r'(?:(?<![a-zA-Z\(])[xXB](?![a-ce-oq-zA-CE-OQ-Z,\.\/])|(?<![:])[:=|](?![\.])|(?<![%#\d])[%#](?![%#\d])|(?<![\d\$])[$](?![\d\.,\$])|[;](?!\()|(?<![\d\(\-\+])8(?![\da-ce-zA-CE-Z\\/])|\*(?![\*\d,.]))',
    # optional tears
    r"(?:['\",])?",
    # optional nose
    r'(?:(?<![\w*])[oc](?![a-zA-Z])|(?:[-‑^]))?',
    # mouth
    r'(?:[(){}\[\]<>|/\\]+|[Þ×þ]|(?<!\d)[30](?!\d)|(?<![\d\*])[*,.@#&](?![\*\d,.])|(?<![\d\$])[$](?![\d\.,\$])|[DOosSJLxXpPbc](?![a-zA-Z]))',
]

_rtl_emoticon = [
    r'(?<![\w])',
    r'(?:[(){}\[\]<>|/\\]+|(?<![\d\.\,])[0](?![\d\.])|(?![\d\*,.@#&])[*,.@#&]|[$]|(?<![a-zA-Z])[DOosSxX])',
    # mouth
    r'(?:[-‑^])?',  # optional nose
    r"(?:['\",])?",  # optional tears
    r'(?:[xX]|[:=|]|[%#]|[$8](?![\d\.])|[;]|\*)',  # eyes
    r'(?:[O]|[0]|[|{><=])?',  # optional hat
    r'(?![a-zA-Z])',
]

_LTR_FACE = ''.join(_ltr_emoticon)
_RTL_FACE = ''.join(_rtl_emoticon)

_short_date = r'\b(?:[12][0-9]{3}[-/\.](?:0?[1-9]|1[0-2])[-/\.](?:0?[1-9]|[12][0-9]|3[01])|' \
              r'(?:0?[1-9]|[12][0-9]|3[01])[-/\.](?:0?[1-9]|1[0-2])[-/\.][12][0-9]{3})\b'
_full_date_parts = [
    # prefix
    r'(?:(?<!:)\b\'?\d{1,4},? ?)',
    r'\b(?:[Jj]an(?:uari)?|[Ff]eb(?:ruari)?|[Mm]a(?:c)?|[Aa]pr(?:il)?|[Mm]ei|[Jj]u(?:n)?|[Jj]ula(?:i)?|[Aa]ug(?:ust)?|[Oo]gos|[Ss]ept?(?:ember)?|[Oo]kt(?:ober)?|[Nn]ov(?:ember)?|[Dd]is(?:ember)?)\b',
    r'(?:(?:,? ?\'?)?\d{1,4}(?:st|nd|rd|n?th)?\b[,\/]? ?\'?\d{2,4}[a-zA-Z]*(?: ?- ?\d{2,4}[a-zA-Z]*)?(?!:\d{1,4})\b)'
]
_fd1 = '(?:{})'.format(
    ''.join(
        [_full_date_parts[0] + '?', _full_date_parts[1], _full_date_parts[2]]
    )
)

_fd2 = '(?:{})'.format(
    ''.join(
        [_full_date_parts[0], _full_date_parts[1], _full_date_parts[2]]
    )
)

_day_month_date = r'\b[0123]?[0-9]\s+(?:[Jj]an(?:uari)?|[Ff]eb(?:ruari)?|[Mm]a(?:c)?|[Aa]pr(?:il)?|[Mm]ei|[Jj]u(?:n)?|[Jj]ula(?:i)?|[Aa]ug(?:ust)?|[Oo]gos|[Ss]ept?(?:ember)?|[Oo]kt(?:ober)?|[Nn]ov(?:ember)?|[Dd]is(?:ember)?)\b'
_month_day_date = r'\b(?:[Jj]an(?:uari)?|[Ff]eb(?:ruari)?|[Mm]a(?:c)?|[Aa]pr(?:il)?|[Mm]ei|[Jj]u(?:n)?|[Jj]ula(?:i)?|[Aa]ug(?:ust)?|[Oo]gos|[Ss]ept?(?:ember)?|[Oo]kt(?:ober)?|[Nn]ov(?:ember)?|[Dd]is(?:ember)?)\s+[0123]?[0-9]\b'
_day_month_year = (
    r'\b[0123]?[0-9]\s+'
    r'(?:[Jj]an(?:uari)?|[Ff]eb(?:ruari)?|[Mm]a(?:c)?|[Aa]pr(?:il)?|[Mm]ei|'
    r'[Jj]u(?:n)?|[Jj]ula(?:i)?|[Aa]ug(?:ust)?|[Oo]gos|[Ss]ept?(?:ember)?|'
    r'[Oo]kt(?:ober)?|[Nn]ov(?:ember)?|[Dd]is(?:ember)?)\s+'
    r'(?:\d{2,4})\b'
)
_date = '(?:' + '(?:' + _fd1 + '|' + _fd2 + ')' + '|' + _short_date + '|' + _day_month_year + '|' + _day_month_date + '|' + _month_day_date + ')'
_time = r'(?:(?:\d+)?\.?\d+\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.|pagi|pgi|morning|tengahari|tngahari|petang|ptg|malam|jam|hours|hour|hrs))|(?:(?:[0-2]?[0-9]|[2][0-3]):(?:[0-5][0-9])(?::(?:[0-5][0-9]))?(?: ?(?:AM|PM|am|pm|a\.m\.|p\.m\.|pagi|pgi|morning|tengahari|tngahari|petang|ptg|malam|hours|hrs|jam))?)'
_today_time = (
    r'(?:(?:pkul|pukul|kul)\s*'
    r'(?:[0-2]?[0-9](?::[0-5][0-9])?(?::[0-5][0-9])?|(?:\d+)?\.?\d+)'
    r'(?:\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.|pagi|pgi|morning|tengahari|tngahari|'
    r'petang|ptg|malam|hours|hrs|jam))?)'
)
_past_date_string = '(?:\\s|\\d+)\\s*(?:minggu|bulan|tahun|hari|thun|hri|mnggu|jam|minit|saat)\\s*(?:lalu|lepas|lps)\\b'
_now_date_string = '(?:sekarang|skrg|jam|tahun|thun|saat|minit) (?:ini|ni)\\b'
_yesterday_tomorrow_date_string = (
    '(?:yesterday|semalam|kelmarin|smalam|esok|esk)\\b'
)
_future_date_string = '(?:dlm|dalam)\\s*\\d+(?:minggu|bulan|tahun|hari|thun|hri|mnggu|jam|minit|saat)\\b'
_depan_date_string = '(?:\\s|\\d+)\\s*(?:minggu|bulan|tahun|hari|thun|hri|mnggu|jam|minit|saat)\\s*(?:depan|dpan|dpn)\\b'

_number = r"(?<!\w)-?\d+(?:[\.,']\d+)?(?!\w)"
_number_with_shortform = r"\b(?:\d+(?:[\.,']\d+)?(?:[KkMmBbJj])|\d+(?:[\.,']\d+)?\s+(?:[Rr]ibu|[Tt]housand|[Jj]uta|[Mm]illion|[Bb]ilion|[Bb]illion))\b"
_percentage = _number + '%'
_money = r"(?:(?:[$€£¢]|RM|rm)\s*\d+(?:[\.,']\d+)?\s*(?:[Rr]ibu|[Jj]uta|[Tt]housand|[Mm]illion|[MmKkBbj](?:n|(?:i(?:lion|llion)?))?)?)\b|(?:\d+(?:[\.,']\d+)?\s*(?:[MmKkBbj](?:n|(?:i(?:lion|llion)?))?|[Rr]ibu|[Jj]uta|[Tt]housand|[Mm]illion)?\s*(?:[$€£¢]|sen|ringgit|cent|penny))\b"
_temperature = "-?\\d+(?:[\\.,']\\d+)?\\s*(?:K|Kelvin|kelvin|Kvin|F|f|Farenheit|farenheit|C|c|Celcius|celcius|clcius|celsius)\\b"
_distance = "-?\\d+(?:[\\.,']\\d+)?\\s*(?:kaki|mtrs|metres|meters|feet|km|m|cm|feet|feets|miles|batu|inch|inches|feets)\\b"
_volume = "-?\\d+(?:[\\.,']\\d+)?\\s*(?:ml|ML|l|L|mililiter|Mililiter|millilitre|liter|litre|litres|liters|gallon|gallons|galon)\\b"
_duration = '\\d+\\s*(?:jam|minit|hari|minggu|bulan|tahun|hours|hour|saat|second|month|months)\\b|(?:sejam|sehari|setahun|sesaat|seminit|sebulan)\\b'
_weight = "\\d+(?:[\\.,']\\d+)?\\s*(?:kg|kilo|kilogram|g|gram|KG)\\b"
_data_size = (
    r'\d+(?:\.\d+)?\s*(?:'
    r'bits?|Bits?|BITs?|'
    r'bit|Bit|BIT|'
    r'bytes?|Bytes?|BYTES?|'
    r'bait|Bait|BAIT|'
    r'kb|Kb|KB|kB|'
    r'kilobytes?|Kilobytes?|KILOBYTES?|'
    r'kilobait|Kilobait|KILOBAIT|'
    r'kilobit|Kilobit|KILOBIT|'
    r'mb|Mb|MB|mB|'
    r'megabytes?|Megabytes?|MEGABYTES?|'
    r'megabait|Megabait|MEGABAIT|'
    r'megabit|Megabit|MEGABIT|'
    r'gb|Gb|GB|gB|'
    r'gigabytes?|Gigabytes?|GIGABYTES?|'
    r'gigabait|Gigabait|GIGABAIT|'
    r'gigabit|Gigabit|GIGABIT|'
    r'tb|Tb|TB|tB|'
    r'terabytes?|Terabytes?|TERABYTES?|'
    r'terabait|Terabait|TERABAIT|'
    r'terabit|Terabit|TERABIT'
    r')\b'
)
_hijri_year = r'\b\d{3,4}\s*[Hh]\b'
_hari_bulan = r'\b(?:[1-9]|[12][0-9]|3[01])[Hh][Bb]\b'
_pada_tarikh = r"\b((?:pada|tarikh)\s+(?:0?[1-9]|[12][0-9]|3[01])\s(?:0?[1-9]|1[0-2]))\b"
_word_dash = r'(?:[A-Za-z0-9]+-){2,}[A-Za-z0-9]+'
_passport = r'^(?:[A-PR-WY][1-9]\d\s?\d{4}[1-9]|[A-Za-z][0-9]{8}|[A-Za-z](?=.*\d)[A-Za-z0-9]{5,19})$'

_left_datetime = '(%s) (%s)' % (_time, _date)
_right_datetime = '(%s) (%s)' % (_date, _time)
_left_datetodaytime = '(%s) (%s)' % (_today_time, _date)
_right_datetodaytime = '(%s) (%s)' % (_date, _today_time)
_left_yesterdaydatetime = '(%s) (%s)' % (_time, _yesterday_tomorrow_date_string)
_right_yesterdaydatetime = '(%s) (%s)' % (
    _yesterday_tomorrow_date_string,
    _time,
)
_left_yesterdaydatetodaytime = '(%s) (%s)' % (
    _today_time,
    _yesterday_tomorrow_date_string,
)
_right_yesterdaydatetodaytime = '(%s) (%s)' % (
    _yesterday_tomorrow_date_string,
    _today_time,
)

_expressions = {
    'hashtag': r'\#\b[\w\-\_]+\b',
    'cashtag': r'(?<![A-Z])\$[A-Z]+\b',
    'tag': r'<[\/]?\w+[\/]?>',
    'user': r'\@\w+',
    'emphasis': r'(?:\*\b\w+\b\*)',
    'censored': r'(?:\b\w+\*+\w+\b)',
    'acronym': r'\b(?:[A-Z]\.)(?:[A-Z]\.)+(?:\.(?!\.))?(?:[A-Z]\b)?',
    'elongated': r'\b[A-Za-z]*([a-zA-Z])\1\1[A-Za-z]*\b',
    'rtl_face': _RTL_FACE,
    'ltr_face': _LTR_FACE,
    'eastern_emoticons': r'(?<![\w])(?:(?:[<>]?[\^;][\W_m][\;^][;<>]?)|(?:[^\s()]?m?[\(][\W_oTOJ]{1,3}[\s]?[\W_oTOJ]{1,3}[)]m?[^\s()]?)|(?:\*?[v>\-\/\\][o0O\_\.][v\-<\/\\]\*?)|(?:[oO0>][\-_\/oO\.\\]{1,2}[oO0>])|(?:\^\^))(?![\w])',
    'rest_emoticons': r'(?<![A-Za-z0-9/()])(?:(?:\^5)|(?:\<3))(?![[A-Za-z0-9/()])',
    # from https://github.com/mathiasbynens/emoji-regex/blob/master/text.js
    'emoji': r'(?:\uD83C\uDFF4\uDB40\uDC67\uDB40\uDC62(?:\uDB40\uDC65\uDB40\uDC6E\uDB40\uDC67|\uDB40\uDC77\uDB40\uDC6C\uDB40\uDC73|\uDB40\uDC73\uDB40\uDC63\uDB40\uDC74)\uDB40\uDC7F|\uD83D\uDC69\u200D\uD83D\uDC69\u200D(?:\uD83D\uDC66\u200D\uD83D\uDC66|\uD83D\uDC67\u200D(?:\uD83D[\uDC66\uDC67]))|\uD83D\uDC68(?:\u200D(?:\u2764\uFE0F\u200D(?:\uD83D\uDC8B\u200D)?\uD83D\uDC68|(?:\uD83D[\uDC68\uDC69])\u200D(?:\uD83D\uDC66\u200D\uD83D\uDC66|\uD83D\uDC67\u200D(?:\uD83D[\uDC66\uDC67]))|\uD83D\uDC66\u200D\uD83D\uDC66|\uD83D\uDC67\u200D(?:\uD83D[\uDC66\uDC67])|[\u2695\u2696\u2708]\uFE0F|\uD83C[\uDF3E\uDF73\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92])|(?:\uD83C[\uDFFB-\uDFFF])\u200D[\u2695\u2696\u2708]\uFE0F|(?:\uD83C[\uDFFB-\uDFFF])\u200D(?:\uD83C[\uDF3E\uDF73\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]))|\uD83D\uDC69\u200D(?:\u2764\uFE0F\u200D(?:\uD83D\uDC8B\u200D(?:\uD83D[\uDC68\uDC69])|\uD83D[\uDC68\uDC69])|\uD83C[\uDF3E\uDF73\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92])|\uD83D\uDC69\u200D\uD83D\uDC66\u200D\uD83D\uDC66|(?:\uD83D\uDC41\uFE0F\u200D\uD83D\uDDE8|\uD83D\uDC69(?:\uD83C[\uDFFB-\uDFFF])\u200D[\u2695\u2696\u2708]|(?:(?:\u26F9|\uD83C[\uDFCB\uDFCC]|\uD83D\uDD75)\uFE0F|\uD83D\uDC6F|\uD83E[\uDD3C\uDDDE\uDDDF])\u200D[\u2640\u2642]|(?:\u26F9|\uD83C[\uDFCB\uDFCC]|\uD83D\uDD75)(?:\uD83C[\uDFFB-\uDFFF])\u200D[\u2640\u2642]|(?:\uD83C[\uDFC3\uDFC4\uDFCA]|\uD83D[\uDC6E\uDC71\uDC73\uDC77\uDC81\uDC82\uDC86\uDC87\uDE45-\uDE47\uDE4B\uDE4D\uDE4E\uDEA3\uDEB4-\uDEB6]|\uD83E[\uDD26\uDD37-\uDD39\uDD3D\uDD3E\uDDD6-\uDDDD])(?:(?:\uD83C[\uDFFB-\uDFFF])\u200D[\u2640\u2642]|\u200D[\u2640\u2642])|\uD83D\uDC69\u200D[\u2695\u2696\u2708])\uFE0F|\uD83D\uDC69\u200D\uD83D\uDC67\u200D(?:\uD83D[\uDC66\uDC67])|\uD83D\uDC69\u200D\uD83D\uDC69\u200D(?:\uD83D[\uDC66\uDC67])|\uD83D\uDC68(?:\u200D(?:(?:\uD83D[\uDC68\uDC69])\u200D(?:\uD83D[\uDC66\uDC67])|\uD83D[\uDC66\uDC67])|\uD83C[\uDFFB-\uDFFF])|\uD83C\uDFF3\uFE0F\u200D\uD83C\uDF08|\uD83D\uDC69\u200D\uD83D\uDC67|\uD83D\uDC69(?:\uD83C[\uDFFB-\uDFFF])\u200D(?:\uD83C[\uDF3E\uDF73\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92])|\uD83D\uDC69\u200D\uD83D\uDC66|\uD83C\uDDF4\uD83C\uDDF2|\uD83C\uDDFD\uD83C\uDDF0|\uD83C\uDDF6\uD83C\uDDE6|\uD83D\uDC69(?:\uD83C[\uDFFB-\uDFFF])|\uD83C\uDDFC(?:\uD83C[\uDDEB\uDDF8])|\uD83C\uDDEB(?:\uD83C[\uDDEE-\uDDF0\uDDF2\uDDF4\uDDF7])|\uD83C\uDDE9(?:\uD83C[\uDDEA\uDDEC\uDDEF\uDDF0\uDDF2\uDDF4\uDDFF])|\uD83C\uDDE7(?:\uD83C[\uDDE6\uDDE7\uDDE9-\uDDEF\uDDF1-\uDDF4\uDDF6-\uDDF9\uDDFB\uDDFC\uDDFE\uDDFF])|\uD83C\uDDF1(?:\uD83C[\uDDE6-\uDDE8\uDDEE\uDDF0\uDDF7-\uDDFB\uDDFE])|\uD83C\uDDFE(?:\uD83C[\uDDEA\uDDF9])|\uD83C\uDDF9(?:\uD83C[\uDDE6\uDDE8\uDDE9\uDDEB-\uDDED\uDDEF-\uDDF4\uDDF7\uDDF9\uDDFB\uDDFC\uDDFF])|\uD83C\uDDF5(?:\uD83C[\uDDE6\uDDEA-\uDDED\uDDF0-\uDDF3\uDDF7-\uDDF9\uDDFC\uDDFE])|\uD83C\uDDEF(?:\uD83C[\uDDEA\uDDF2\uDDF4\uDDF5])|\uD83C\uDDED(?:\uD83C[\uDDF0\uDDF2\uDDF3\uDDF7\uDDF9\uDDFA])|\uD83C\uDDEE(?:\uD83C[\uDDE8-\uDDEA\uDDF1-\uDDF4\uDDF6-\uDDF9])|\uD83C\uDDFB(?:\uD83C[\uDDE6\uDDE8\uDDEA\uDDEC\uDDEE\uDDF3\uDDFA])|\uD83C\uDDEC(?:\uD83C[\uDDE6\uDDE7\uDDE9-\uDDEE\uDDF1-\uDDF3\uDDF5-\uDDFA\uDDFC\uDDFE])|\uD83C\uDDF7(?:\uD83C[\uDDEA\uDDF4\uDDF8\uDDFA\uDDFC])|\uD83C\uDDEA(?:\uD83C[\uDDE6\uDDE8\uDDEA\uDDEC\uDDED\uDDF7-\uDDFA])|\uD83C\uDDFA(?:\uD83C[\uDDE6\uDDEC\uDDF2\uDDF3\uDDF8\uDDFE\uDDFF])|\uD83C\uDDE8(?:\uD83C[\uDDE6\uDDE8\uDDE9\uDDEB-\uDDEE\uDDF0-\uDDF5\uDDF7\uDDFA-\uDDFF])|\uD83C\uDDE6(?:\uD83C[\uDDE8-\uDDEC\uDDEE\uDDF1\uDDF2\uDDF4\uDDF6-\uDDFA\uDDFC\uDDFD\uDDFF])|[#\*0-9]\uFE0F\u20E3|\uD83C\uDDF8(?:\uD83C[\uDDE6-\uDDEA\uDDEC-\uDDF4\uDDF7-\uDDF9\uDDFB\uDDFD-\uDDFF])|\uD83C\uDDFF(?:\uD83C[\uDDE6\uDDF2\uDDFC])|\uD83C\uDDF0(?:\uD83C[\uDDEA\uDDEC-\uDDEE\uDDF2\uDDF3\uDDF5\uDDF7\uDDFC\uDDFE\uDDFF])|\uD83C\uDDF3(?:\uD83C[\uDDE6\uDDE8\uDDEA-\uDDEC\uDDEE\uDDF1\uDDF4\uDDF5\uDDF7\uDDFA\uDDFF])|\uD83C\uDDF2(?:\uD83C[\uDDE6\uDDE8-\uDDED\uDDF0-\uDDFF])|(?:\uD83C[\uDFC3\uDFC4\uDFCA]|\uD83D[\uDC6E\uDC71\uDC73\uDC77\uDC81\uDC82\uDC86\uDC87\uDE45-\uDE47\uDE4B\uDE4D\uDE4E\uDEA3\uDEB4-\uDEB6]|\uD83E[\uDD26\uDD37-\uDD39\uDD3D\uDD3E\uDDD6-\uDDDD])(?:\uD83C[\uDFFB-\uDFFF])|(?:\u26F9|\uD83C[\uDFCB\uDFCC]|\uD83D\uDD75)(?:\uD83C[\uDFFB-\uDFFF])|(?:[\u261D\u270A-\u270D]|\uD83C[\uDF85\uDFC2\uDFC7]|\uD83D[\uDC42\uDC43\uDC46-\uDC50\uDC66\uDC67\uDC70\uDC72\uDC74-\uDC76\uDC78\uDC7C\uDC83\uDC85\uDCAA\uDD74\uDD7A\uDD90\uDD95\uDD96\uDE4C\uDE4F\uDEC0\uDECC]|\uD83E[\uDD18-\uDD1C\uDD1E\uDD1F\uDD30-\uDD36\uDDD1-\uDDD5])(?:\uD83C[\uDFFB-\uDFFF])|(?:[\u261D\u26F9\u270A-\u270D]|\uD83C[\uDF85\uDFC2-\uDFC4\uDFC7\uDFCA-\uDFCC]|\uD83D[\uDC42\uDC43\uDC46-\uDC50\uDC66-\uDC69\uDC6E\uDC70-\uDC78\uDC7C\uDC81-\uDC83\uDC85-\uDC87\uDCAA\uDD74\uDD75\uDD7A\uDD90\uDD95\uDD96\uDE45-\uDE47\uDE4B-\uDE4F\uDEA3\uDEB4-\uDEB6\uDEC0\uDECC]|\uD83E[\uDD18-\uDD1C\uDD1E\uDD1F\uDD26\uDD30-\uDD39\uDD3D\uDD3E\uDDD1-\uDDDD])(?:\uD83C[\uDFFB-\uDFFF])?|(?:[\u231A\u231B\u23E9-\u23EC\u23F0\u23F3\u25FD\u25FE\u2614\u2615\u2648-\u2653\u267F\u2693\u26A1\u26AA\u26AB\u26BD\u26BE\u26C4\u26C5\u26CE\u26D4\u26EA\u26F2\u26F3\u26F5\u26FA\u26FD\u2705\u270A\u270B\u2728\u274C\u274E\u2753-\u2755\u2757\u2795-\u2797\u27B0\u27BF\u2B1B\u2B1C\u2B50\u2B55]|\uD83C[\uDC04\uDCCF\uDD8E\uDD91-\uDD9A\uDDE6-\uDDFF\uDE01\uDE1A\uDE2F\uDE32-\uDE36\uDE38-\uDE3A\uDE50\uDE51\uDF00-\uDF20\uDF2D-\uDF35\uDF37-\uDF7C\uDF7E-\uDF93\uDFA0-\uDFCA\uDFCF-\uDFD3\uDFE0-\uDFF0\uDFF4\uDFF8-\uDFFF]|\uD83D[\uDC00-\uDC3E\uDC40\uDC42-\uDCFC\uDCFF-\uDD3D\uDD4B-\uDD4E\uDD50-\uDD67\uDD7A\uDD95\uDD96\uDDA4\uDDFB-\uDE4F\uDE80-\uDEC5\uDECC\uDED0-\uDED2\uDEEB\uDEEC\uDEF4-\uDEF8]|\uD83E[\uDD10-\uDD3A\uDD3C-\uDD3E\uDD40-\uDD45\uDD47-\uDD4C\uDD50-\uDD6B\uDD80-\uDD97\uDDC0\uDDD0-\uDDE6])|(?:[#\*0-9\xA9\xAE\u203C\u2049\u2122\u2139\u2194-\u2199\u21A9\u21AA\u231A\u231B\u2328\u23CF\u23E9-\u23F3\u23F8-\u23FA\u24C2\u25AA\u25AB\u25B6\u25C0\u25FB-\u25FE\u2600-\u2604\u260E\u2611\u2614\u2615\u2618\u261D\u2620\u2622\u2623\u2626\u262A\u262E\u262F\u2638-\u263A\u2640\u2642\u2648-\u2653\u2660\u2663\u2665\u2666\u2668\u267B\u267F\u2692-\u2697\u2699\u269B\u269C\u26A0\u26A1\u26AA\u26AB\u26B0\u26B1\u26BD\u26BE\u26C4\u26C5\u26C8\u26CE\u26CF\u26D1\u26D3\u26D4\u26E9\u26EA\u26F0-\u26F5\u26F7-\u26FA\u26FD\u2702\u2705\u2708-\u270D\u270F\u2712\u2714\u2716\u271D\u2721\u2728\u2733\u2734\u2744\u2747\u274C\u274E\u2753-\u2755\u2757\u2763\u2764\u2795-\u2797\u27A1\u27B0\u27BF\u2934\u2935\u2B05-\u2B07\u2B1B\u2B1C\u2B50\u2B55\u3030\u303D\u3297\u3299]|\uD83C[\uDC04\uDCCF\uDD70\uDD71\uDD7E\uDD7F\uDD8E\uDD91-\uDD9A\uDDE6-\uDDFF\uDE01\uDE02\uDE1A\uDE2F\uDE32-\uDE3A\uDE50\uDE51\uDF00-\uDF21\uDF24-\uDF93\uDF96\uDF97\uDF99-\uDF9B\uDF9E-\uDFF0\uDFF3-\uDFF5\uDFF7-\uDFFF]|\uD83D[\uDC00-\uDCFD\uDCFF-\uDD3D\uDD49-\uDD4E\uDD50-\uDD67\uDD6F\uDD70\uDD73-\uDD7A\uDD87\uDD8A-\uDD8D\uDD90\uDD95\uDD96\uDDA4\uDDA5\uDDA8\uDDB1\uDDB2\uDDBC\uDDC2-\uDDC4\uDDD1-\uDDD3\uDDDC-\uDDDE\uDDE1\uDDE3\uDDE8\uDDEF\uDDF3\uDDFA-\uDE4F\uDE80-\uDEC5\uDECB-\uDED2\uDEE0-\uDEE5\uDEE9\uDEEB\uDEEC\uDEF0\uDEF3-\uDEF8]|\uD83E[\uDD10-\uDD3A\uDD3C-\uDD3E\uDD40-\uDD45\uDD47-\uDD4C\uDD50-\uDD6B\uDD80-\uDD97\uDDC0\uDDD0-\uDDE6])\uFE0F?)',
    # "EMOJI": r"(?:[\u00A9\u00AE\u203C\u2049\u2122\u2139\u2194-\u2199\u21A9-\u21AA\u231A-\u231B\u2328\u23CF\u23E9-\u23F3\u23F8-\u23FA\u24C2\u25AA-\u25AB\u25B6\u25C0\u25FB-\u25FE\u2600-\u2604\u260E\u2611\u2614-\u2615\u2618\u261D\u2620\u2622-\u2623\u2626\u262A\u262E-\u262F\u2638-\u263A\u2640\u2642\u2648-\u2653\u2660\u2663\u2665-\u2666\u2668\u267B\u267F\u2692-\u2697\u2699\u269B-\u269C\u26A0-\u26A1\u26AA-\u26AB\u26B0-\u26B1\u26BD-\u26BE\u26C4-\u26C5\u26C8\u26CE-\u26CF\u26D1\u26D3-\u26D4\u26E9-\u26EA\u26F0-\u26F5\u26F7-\u26FA\u26FD\u2702\u2705\u2708-\u270D\u270F\u2712\u2714\u2716\u271D\u2721\u2728\u2733-\u2734\u2744\u2747\u274C\u274E\u2753-\u2755\u2757\u2763-\u2764\u2795-\u2797\u27A1\u27B0\u27BF\u2934-\u2935\u2B05-\u2B07\u2B1B-\u2B1C\u2B50\u2B55\u3030\u303D\u3297\u3299]|(?:\uD83C[\uDC04\uDCCF\uDD70-\uDD71\uDD7E-\uDD7F\uDD8E\uDD91-\uDD9A\uDDE6-\uDDFF\uDE01-\uDE02\uDE1A\uDE2F\uDE32-\uDE3A\uDE50-\uDE51\uDF00-\uDF21\uDF24-\uDF93\uDF96-\uDF97\uDF99-\uDF9B\uDF9E-\uDFF0\uDFF3-\uDFF5\uDFF7-\uDFFF]|\uD83D[\uDC00-\uDCFD\uDCFF-\uDD3D\uDD49-\uDD4E\uDD50-\uDD67\uDD6F-\uDD70\uDD73-\uDD7A\uDD87\uDD8A-\uDD8D\uDD90\uDD95-\uDD96\uDDA4-\uDDA5\uDDA8\uDDB1-\uDDB2\uDDBC\uDDC2-\uDDC4\uDDD1-\uDDD3\uDDDC-\uDDDE\uDDE1\uDDE3\uDDE8\uDDEF\uDDF3\uDDFA-\uDE4F\uDE80-\uDEC5\uDECB-\uDED2\uDEE0-\uDEE5\uDEE9\uDEEB-\uDEEC\uDEF0\uDEF3-\uDEF6]|\uD83E[\uDD10-\uDD1E\uDD20-\uDD27\uDD30\uDD33-\uDD3A\uDD3C-\uDD3E\uDD40-\uDD45\uDD47-\uDD4B\uDD50-\uDD5E\uDD80-\uDD91\uDDC0]))",
    'quotes': r'\"(\\.|[^\"]){2,}\"',
    'percent': _percentage,
    'repeat_puncts': r'([!?.]){2,}',
    'money': _money,
    'email': r'(?:^|(?<=[^\w@.)]))(?:[\w+-](?:\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(?:\.(?:[a-z]{2,})){1,3}(?:$|(?=\b))',
    'phone': r'(?<![0-9])(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![0-9])',
    # 'temperature': _temperature,
    # 'distance': _distance,
    'number': _number,
    'number_with_shortform': _number_with_shortform,
    'allcaps': r'(?<![#@$])\b([A-Z][A-Z ]{1,}[A-Z])\b',
    'url': r'(?:https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})',
    # https://regexr.com/39nr7
    'url_v2': r'[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)',
    # https://gist.github.com/dperini/729294
    'url_dperini': r'^(?:(?:(?:https?|ftp):)?\/\/)(?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z0-9\u00a1-\uffff][a-z0-9\u00a1-\uffff_-]{0,62})?[a-z0-9\u00a1-\uffff]\.)+(?:[a-z\u00a1-\uffff]{2,}\.?))(?::\d{2,5})?(?:[/?#]\S*)?$',
    'date': _date,
    'time': _time,
    'time_pukul': _today_time,
    # "CAMEL_SPLIT": '((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))',
    # r'((?<=[a-z])[A-Z]|(?<=[A-Z][A-Z])[a-z]|(?<!^)(?<![A-Z])[A-Z](?=[a-z])|[0-9]+|(?<=[0-9\-\_])[A-Za-z]|[\-\_])',
    'camel_split': r'((?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])|[0-9]+|(?<=[0-9\-\_])[A-Za-z]|[\-\_])',
    # REGEX_NORMALIZE_ELONG = '(.)\1+')
    'normalize_elong': r'(.)\1{2,}',
    'normalize_elong1': r'(.)\1{1,}',
    'word': r'(?:[\w_]+)',
    'hypen': r'\w+(?:-\w+)+',
    'apostrophe': r'\w+\'(?:s)?',
    'temperature': _temperature,
    'distance': _distance,
    'volume': _volume,
    'duration': _duration,
    'weight': _weight,
    'data_size': _data_size,
    'ic': r'(([[0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01]))-([0-9]{2})-([0-9]{4})',
    'bracket': r'(\(.*?\))',
    'title': r'Sdn\.|Bhd\.|Corp\.|Corporation\.|corp\.|Datuk\.|datuk\.|Datin.\|datin.\|Datik\.|datik\.|dr\.|Dr\.|DR\.|yb\.|YB\.|hj\.|HJ\.|Hj\.|ybm\.|YBM\.|Ybm\.|tyt\.|TYT\.|yab\.|YAB\.|Yab\.|ybm\.|YBM\.|Ybm\.|yabhg\.|YABHG.\|Yabhg\.|ybhg\.|YBHG\.|Ybhg\.|YBhg\.|phd\.|PhD\.',
    'parliament': r'[A-Z]\.\d+',
    'hijri_year': _hijri_year,
    'hari_bulan': _hari_bulan,
    'pada_tarikh': _pada_tarikh,
    'word_dash': _word_dash,
    'passport': _passport,
}
