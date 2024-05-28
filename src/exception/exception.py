class CustomExceptionHandler:
    def __init__(self, exception, sys_module):
        self.exception = exception

        _, _, tb = sys_module.exc_info()

        self.line_number = tb.tb_lineno
        self.filename = tb.tb_frame.f_code.co_filename
        self.error_message = str(exception)

    def __str__(self):
        return (
            f"Error: {self.error_message} in {self.filename} at line {self.line_number}"
        )

    def __repr__(self):
        return (
            f"Error: {self.error_message} in {self.filename} at line {self.line_number}"
        )


# Path: src/exception/my_exception.py
