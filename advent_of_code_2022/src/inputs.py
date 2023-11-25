def read_input(day: int) -> str:
    """Load Input for Given Day

    Args:
        day (int): Which day to load

    Returns:
        str: input for that day's puzzle
    """

    with open(file=f"input/day_{day}.txt", mode="r", encoding="utf-8") as file:
        input_str = file.read()

    return input_str
