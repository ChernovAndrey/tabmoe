from enum import Enum
from typing import Type, TypeVar, Optional

T = TypeVar("T", bound=Enum)


def validate_enum(enum_class: Type[T], value: str) -> Optional[T]:
    """
    Validates whether a given value (case-insensitive) is a valid member of the specified Enum.

    Args:
        enum_class (Type[T]): The Enum class to validate against.
        value (str): The value to check.

    Returns:
        Optional[T]: The corresponding Enum member if valid, else None.

    Prints:
        Error message with valid options if the value is invalid.
    """
    value = value.lower()  # Convert input to lowercase for case insensitivity
    enum_map = {e.value.lower(): e for e in enum_class}  # Create case-insensitive mapping

    if value in enum_map:
        return enum_map[value]

    valid_values = ", ".join(e.value for e in enum_class)
    print(f"❌ '{value}' is NOT a valid {enum_class.__name__}. Valid options: {valid_values}")
    return None
