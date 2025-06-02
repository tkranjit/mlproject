from src.logger import logging
import sys


# Configure logging to output to console and file

def custom_exception(error,error_details:sys):
    """
    Custom exception class to handle exceptions with detailed error messages.
    """
    _,_,exc_tb=error_details.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename
    line_number=exc_tb.tb_lineno
    error_message="{0} occurred in script: {1} at line number: {2}".format(filename,line_number,str(error))
    return error_message

class CustomException(Exception):
    """
    Custom exception class that inherits from the built-in Exception class.
    It provides a custom error message when an exception is raised.
    """
    def __init__(self,error,error_details:sys):
        super().__init__(custom_exception(error,error_details))
        self.error_message=custom_exception(error,error_details)
    
    def __str__(self):
        return self.error_message


