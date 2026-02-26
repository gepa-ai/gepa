import pytest # noqa    
 
from kernelbench.utils import extract_first_code, extract_code_blocks, extract_last_code

def check_code_assertions(code: str, expected_code: str):
    """
    Check code is equivalent (don't worry about whitespace)
    """
    if code is None:
        assert expected_code == ""
    else:
        assert code.replace("\n", "").replace(" ", "") == expected_code.replace("\n", "").replace(" ", "")


def test_extract_last_code():
    # Test with Python code block
    example_output = """The LLM wrote some code here
    ```python
    def hello():
        print("Hello")
    ```
    and it says more stuff afterwards"""
    code = extract_last_code(example_output, ["python", "cpp"])
    check_code_assertions(code, "def hello():\n    print(\"Hello\")")


    example_output = """The LLM wrote some code here
    ```cpp
    int main() {
        return 0;
    }
    ```

    and some other code block 
    ```python
    def hello():
        print("Hello")
    ``` 
    and it says more stuff afterwards"""
    code = extract_last_code(example_output, ["python", "cpp"])
    check_code_assertions(code, "def hello():\n    print(\"Hello\")")



def test_extract_first_code():
    # Test with Python code block
    example_output = """The LLM wrote some code here
    ```python
    def hello():
        print("Hello")
    ```
    and it says more stuff afterwards"""
    
    code = extract_first_code(example_output, ["python", "cpp"])
    check_code_assertions(code, "def hello():\n    print(\"Hello\")")

    # Test with no code block
    text = "Some code here"
    code = extract_first_code(text, ["python", "cpp"]) 
    check_code_assertions(code, "")

    # Test with empty code block
    text = "```python\n```"
    code = extract_first_code(text, ["python", "cpp"])
    check_code_assertions(code, "")


    # Test with multiple code blocks
    text = """```python
    def hello():
        print("Hello")
    ```

    ```cpp
    int main() {
        return 0;
    }
    ```
    """
    # NOTE: is this a problem 
    code = extract_first_code(text, ["python", "cpp"])
    check_code_assertions(code, "def hello():\n    print(\"Hello\")")
# Test python hash



def test_extract_code_blocks():
    text = """```python
    def hello():
        print("Hello")
    ```
    """
    code = extract_code_blocks(text, ["python", "rust"])
    check_code_assertions(code, "def hello():\n    print(\"Hello\")")

    text = """```python
    def hello():
        print("Hello")
    ```

    ```cpp
    int main() {
        return 0;
    }
    ```
    """
    # NOTE: is this a problem 
    code = extract_code_blocks(text, ["python", "cpp"])
    check_code_assertions(code, "def hello():\n    print(\"Hello\") \n int main() { \n return 0; \n }")

