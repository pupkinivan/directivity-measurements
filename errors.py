class InvalidOctaveDividerError(IOError):
    def __init__(self, message: str):
        self._message = message

    def __str__(self):
        return self._message