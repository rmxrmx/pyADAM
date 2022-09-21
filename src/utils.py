"""
Utility functions used in various parts of the script.
"""

def convert_to_participant(number: int) -> str:
    """
    Function for converting an integer version of the participant
    to a string version.
    """
    if number == 0:
        return "participant_1"
    elif number == 1:
        return "participant_2"
    elif number == 2:
        return "metronome"
    else:
        return "unknown"
        