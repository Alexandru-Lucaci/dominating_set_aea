class TestFile:
    def __init__(self, test_file: str, sol_file: str):
        self.test_file = test_file
        self.sol_file = sol_file

    def get_test_files(self):
        return self.test_files

    def get_sol_files(self):
        return self.sol_files

    def __repr__(self):
        return f"TestFile(test_file={self.test_file}, sol_file={self.sol_file})"
