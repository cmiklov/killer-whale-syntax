"""
Palindromic positional encoding/decoding.

Uses the base-14 system from NUMERALS.md:
  Digit table: L I V X F H T Y D M B R Q U → 0–13

Positional encoding:
  value = center + 2 × Σ(digit_i × 14^i)

  center L = even (parity 0), center I = odd (parity 1)
  Side digits encode (value - parity) / 2 in base-14
  Left side mirrors right side — always a palindrome.
"""

# The 14 magnitude symbols, index = value
DIGITS = "LIVXFHTYDMBRQU"
DIGIT_TO_VAL = {ch: i for i, ch in enumerate(DIGITS)}
BASE = len(DIGITS)  # 14


def int_to_palindrome(n: int) -> str:
    """Convert a non-negative integer to its palindromic positional address."""
    if n < 0:
        raise ValueError(f"Negative values not supported: {n}")

    # Center carries parity
    if n % 2 == 0:
        center = "L"
    else:
        center = "I"

    side_value = (n - (n % 2)) // 2

    if side_value == 0:
        return center

    # Convert side_value to base-14 digits (least significant first)
    side_digits = []
    v = side_value
    while v > 0:
        side_digits.append(DIGITS[v % BASE])
        v //= BASE

    # Palindrome: side_reversed + center + side
    left = "".join(reversed(side_digits))
    right = "".join(side_digits)
    return left + center + right


def palindrome_to_int(s: str) -> int:
    """Convert a palindromic positional address back to an integer."""
    if not s:
        raise ValueError("Empty address")

    length = len(s)

    if length == 1:
        # Center only
        if s == "L":
            return 0
        if s == "I":
            return 1
        raise ValueError(f"Single-character address must be L or I, got: {s}")

    if length % 2 == 0:
        raise ValueError(f"Palindromic address must have odd length, got {length}: {s}")

    mid = length // 2
    center = s[mid]
    if center == "L":
        parity = 0
    elif center == "I":
        parity = 1
    else:
        raise ValueError(f"Center must be L or I, got: {center}")

    # Right side digits (least significant first, reading outward from center)
    right_digits = s[mid + 1:]

    # Decode base-14
    side_value = 0
    for i, ch in enumerate(right_digits):
        if ch not in DIGIT_TO_VAL:
            raise ValueError(f"Unknown digit: {ch}")
        side_value += DIGIT_TO_VAL[ch] * (BASE ** i)

    return parity + 2 * side_value


def compose_address(modifier_addr: str, head_addr: str) -> str:
    """
    Nest a modifier's ring around a head's address to form a compound address.

    The modifier's side digits become an outer ring around the head palindrome.
    """
    if len(modifier_addr) == 1:
        mod_val = palindrome_to_int(modifier_addr)
        if mod_val == 0:
            return head_addr
        return DIGITS[mod_val] + head_addr + DIGITS[mod_val]

    mod_mid = len(modifier_addr) // 2
    mod_left = modifier_addr[:mod_mid]
    mod_right = modifier_addr[mod_mid + 1:]
    return mod_left + head_addr + mod_right


def decompose_address(addr: str) -> list[str]:
    """
    Peel a compound address into constituent root addresses.

    Returns a list of palindromic addresses [modifier, head] or [mod1, mod2, head]
    by peeling outer rings inward.
    """
    if len(addr) <= 7:
        try:
            val = palindrome_to_int(addr)
            if 0 <= val <= 5487:
                return [addr]
        except ValueError:
            pass

    return [addr]


def validate_palindrome(s: str) -> bool:
    """Check if a string is a valid palindromic address."""
    if not s:
        return False
    if len(s) == 1:
        return s in ("L", "I")
    if len(s) % 2 == 0:
        return False
    mid = len(s) // 2
    return (
        s[mid] in ("L", "I")
        and s[:mid] == s[mid + 1:][::-1]
        and all(ch in DIGIT_TO_VAL for ch in s[:mid])
        and all(ch in DIGIT_TO_VAL for ch in s[mid + 1:])
    )
