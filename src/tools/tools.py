from strands import Agent, tool

@tool
def letter_counter(word: str, letter: str) -> int:
    """
    Count occurrences of a specific letter in a word.

    Args:
        word (str): The input word to search in
        letter (str): The specific letter to count

    Returns:
        int: The number of occurrences of the letter in the word
    """
    if len(letter) != 1:
        raise ValueError("The 'letter' parameter must be a single character")

    return word.lower().count(letter.lower())


from strands import tool


@tool
def reverse_string(text: str) -> str:
    """
    Reverse the characters in a given string.

    Args:
        text: The input string to reverse.

    Returns:
        The reversed string.
    """
    return text[::-1]
