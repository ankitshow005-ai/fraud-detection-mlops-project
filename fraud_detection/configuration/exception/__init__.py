import sys
from fraud_detection.configuration.logger import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Creates detailed error message with file name and line number
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        f"\n🔴 Exception Occurred\n"
        f"File      : {file_name}\n"
        f"Line No   : {line_number}\n"
        f"Error     : {str(error)}"
    )

    return error_message


class MyException(Exception):
    """
    Custom project exception
    """
    def __init__(self, error: Exception, error_detail: sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message