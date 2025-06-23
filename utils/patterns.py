import regex

JSON_PATTERN = regex.compile(r"\{(?:[^{}]|(?R))*\}", regex.DOTALL)
UCI_PATTERN = regex.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
