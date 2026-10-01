#!/usr/bin/env python3
"""
computor v1

Solves polynomial equations of degree <= 2.

No math library is used: sqrt is implemented by hand (Newton's method),
and all coefficient arithmetic is done with exact rationals so that
fractions printed as answers are really exact, never rounded guesses.
"""

#!/usr/bin/env python3
"""
computor v1

Solves polynomial equations of degree <= 2.

No math library is used: sqrt is implemented by hand (Newton's method),
and all coefficient arithmetic is done with exact rationals so that
fractions printed as answers are really exact, never rounded guesses.
"""

import sys
import re


# ---------------------------------------------------------------------------
# Exact rational arithmetic (no fractions module, no floats)
# ---------------------------------------------------------------------------

def my_gcd(a, b):
    """Greatest common divisor, iterative Euclid."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


class Rat:
    """An exact rational number num/den, always kept irreducible, den > 0."""

    __slots__ = ("num", "den")

    def __init__(self, num, den=1):
        if den == 0:
            raise ZeroDivisionError("rational with zero denominator")
        if den < 0:
            num, den = -num, -den
        g = my_gcd(num, den) or 1
        self.num = num // g
        self.den = den // g

    # --- the four allowed operations -------------------------------------
    def __add__(self, o):
        return Rat(self.num * o.den + o.num * self.den, self.den * o.den)

    def __sub__(self, o):
        return Rat(self.num * o.den - o.num * self.den, self.den * o.den)

    def __mul__(self, o):
        return Rat(self.num * o.num, self.den * o.den)

    def __truediv__(self, o):
        if o.num == 0:
            raise ZeroDivisionError("division by zero")
        return Rat(self.num * o.den, self.den * o.num)

    def __neg__(self):
        return Rat(-self.num, self.den)

    # --- helpers ----------------------------------------------------------
    def is_zero(self):
        return self.num == 0

    def sign(self):
        return (self.num > 0) - (self.num < 0)

    def to_float(self):
        return self.num / self.den

    def __eq__(self, o):
        return self.num == o.num and self.den == o.den

    def __repr__(self):
        return f"Rat({self.num}/{self.den})"


ZERO = Rat(0)


def rat_from_string(text):
    """'9.3' -> Rat(93, 10) exactly. '-5' -> Rat(-5, 1). No float involved."""
    negative = text.startswith("-")
    text = text.lstrip("+-")
    if "." in text:
        whole, frac = text.split(".")
        whole = whole or "0"
        frac = frac or "0"
        num = int(whole + frac)
        den = 10 ** len(frac)
    else:
        num = int(text)
        den = 1
    r = Rat(num, den)
    return -r if negative else r


# ---------------------------------------------------------------------------
# Square root, implemented by hand
# ---------------------------------------------------------------------------

def isqrt_exact(n):
    """Integer square root of n >= 0, or None if n is not a perfect square."""
    if n < 0:
        return None
    if n < 2:
        return n
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x if x * x == n else None


def rat_sqrt_exact(r):
    """Exact square root of a non-negative Rat, or None if irrational."""
    if r.num < 0:
        return None
    a = isqrt_exact(r.num)
    b = isqrt_exact(r.den)
    if a is None or b is None:
        return None
    return Rat(a, b)


def my_sqrt(x):
    """Float square root by Newton's method (Babylonian). x >= 0."""
    if x < 0:
        raise ValueError("sqrt of a negative number")
    if x == 0:
        return 0.0
    guess = x if x >= 1 else 1.0
    for _ in range(200):
        better = (guess + x / guess) / 2
        if better == guess:
            break
        guess = better
    return guess


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def fmt_decimal(value, places=6):
    s = f"{value:.{places}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def fmt_rat(r):
    if r.den == 1:
        return str(r.num)
    rest, k = r.den, 0
    while rest % 2 == 0:
        rest //= 2
        k += 1
    twos = k
    k = 0
    while rest % 5 == 0:
        rest //= 5
        k += 1
    if rest != 1:
        return f"{r.num}/{r.den}"
    places = twos if twos > k else k
    scaled = r.num * (10 ** places) // r.den
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(places + 1, "0")
    text = f"{sign}{digits[:-places]}.{digits[-places:]}"
    return text.rstrip("0").rstrip(".")


def fmt_solution(r):
    return str(r.num) if r.den == 1 else f"{r.num}/{r.den}"


def fmt_imaginary(r):
    if r.den == 1:
        return f"{r.num}i"
    return f"{r.num}i/{r.den}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

TERM_RE = re.compile(
    r"""^
    (?:(?P<coeff>\d+(?:\.\d*)?|\.\d+)
       (?:\*?(?P<xa>X)(?:\^(?P<pa>\d+))?)?
     |
       (?P<xb>X)(?:\^(?P<pb>\d+))?
    )$""",
    re.VERBOSE | re.IGNORECASE,
)


class ParseError(Exception):
    pass


def parse_side(side):
    side = re.sub(r"\s+", "", side)
    if not side:
        raise ParseError("an empty side is not a valid expression")
    bad = sorted({c for c in side if not re.match(r"[0-9Xx.^*+-]", c)})
    if bad:
        raise ParseError("unexpected character(s): " + " ".join(bad))

    chunks = re.findall(r"[+-]?[^+-]+", side)
    terms = {}
    for chunk in chunks:
        sign = -1 if chunk.startswith("-") else 1
        body = chunk.lstrip("+-")
        if not body:
            raise ParseError("a sign is not followed by a term")
        m = TERM_RE.match(body)
        if not m:
            raise ParseError(f"'{body}' is not a valid term (expected a * X^p)")
        coeff_txt = m.group("coeff")
        coeff = rat_from_string(coeff_txt) if coeff_txt else Rat(1)
        if sign < 0:
            coeff = -coeff
        has_x = m.group("xa") or m.group("xb")
        power = m.group("pa") or m.group("pb")
        exponent = int(power) if power else (1 if has_x else 0)
        terms[exponent] = terms.get(exponent, ZERO) + coeff
    return terms


def parse_equation(text):
    if text.count("=") != 1:
        raise ParseError("the equation must contain exactly one '='")
    left, right = text.split("=")
    return parse_side(left), parse_side(right)


# ---------------------------------------------------------------------------
# Reduction and display
# ---------------------------------------------------------------------------

def reduce_equation(left, right):
    reduced = {}
    for exp in set(left) | set(right):
        reduced[exp] = left.get(exp, ZERO) - right.get(exp, ZERO)
    return reduced


def degree_of(reduced):
    degrees = [e for e, c in reduced.items() if not c.is_zero()]
    return max(degrees) if degrees else 0


def print_reduced_form(reduced):
    # keep zero terms only up to the highest non-null exponent (the subject's
    # own examples keep a "0 * X^2" wedged between real terms, but nothing
    # should trail after the last non-null one)
    top = max((e for e, c in reduced.items() if not c.is_zero()), default=None)
    parts = []
    for exp in sorted(reduced):
        if top is not None and exp > top:
            continue
        coeff = reduced[exp]
        if coeff.sign() < 0:
            sign = "- " if parts else "-"
            shown = -coeff
        else:
            sign = "+ " if parts else ""
            shown = coeff
        parts.append(f"{sign}{fmt_rat(shown)} * X^{exp}")
    if not parts:
        parts = ["0 * X^0"]
    print(f"Reduced form: {' '.join(parts)} = 0")


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

def solve_degree_one(a, b, verbose, want_fraction=False):
    if verbose:
        print(f"Steps: X = -b / a = -({fmt_rat(b)}) / ({fmt_rat(a)})")
    print("The solution is:")
    x = -b / a
    # mandatory part shows a decimal here (subject's own example: -0.25,
    # not -1/4); the fraction form is the bonus, only with --fraction
    print(fmt_solution(x) if want_fraction else fmt_decimal(x.to_float()))


def solve_degree_two(a, b, c, verbose, want_fraction=False):
    two_a = Rat(2) * a
    delta = (b * b) - (Rat(4) * a * c)

    if verbose:
        print(f"Steps: delta = b^2 - 4ac = ({fmt_rat(b)})^2 - 4 * "
              f"({fmt_rat(a)}) * ({fmt_rat(c)}) = {fmt_rat(delta)}")

    def show_real(x):
        # mandatory part prints a decimal for real roots (subject never
        # shows a real root as a fraction); --fraction is the bonus
        return fmt_solution(x) if want_fraction else fmt_decimal(x.to_float())

    if delta.is_zero():
        print("Discriminant is zero, the solution is:")
        print(show_real(-b / two_a))
        return

    if delta.sign() > 0:
        print("Discriminant is strictly positive, the two solutions are:")
        root = rat_sqrt_exact(delta)
        if root is not None:
            print(show_real((-b + root) / two_a))
            print(show_real((-b - root) / two_a))
        else:
            r = my_sqrt(delta.to_float())
            d = two_a.to_float()
            print(fmt_decimal((-b.to_float() + r) / d))
            print(fmt_decimal((-b.to_float() - r) / d))
        return

    print("Discriminant is strictly negative, the two complex solutions are:")
    real = -b / two_a
    root = rat_sqrt_exact(-delta)
    if root is not None:
        imag = root / two_a
        if imag.sign() < 0:
            imag = -imag
        real_s, imag_s = fmt_solution(real), fmt_imaginary(imag)
    else:
        r = my_sqrt(-delta.to_float())
        imag_f = abs(r / two_a.to_float())
        real_s, imag_s = fmt_decimal(real.to_float()), fmt_decimal(imag_f) + "i"
    print(f"{real_s} + {imag_s}")
    print(f"{real_s} - {imag_s}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(equation, verbose=False, want_fraction=False):
    left, right = parse_equation(equation)
    reduced = reduce_equation(left, right)

    print_reduced_form(reduced)
    degree = degree_of(reduced)
    print(f"Polynomial degree: {degree}")

    a = reduced.get(2, ZERO)
    b = reduced.get(1, ZERO)
    c = reduced.get(0, ZERO)

    if degree > 2:
        print("The polynomial degree is strictly greater than 2, I can't solve.")
    elif degree == 2:
        solve_degree_two(a, b, c, verbose, want_fraction)
    elif degree == 1:
        solve_degree_one(b, c, verbose, want_fraction)
    else:
        if c.is_zero():
            print("Any real number is a solution.")
        else:
            print("No solution.")


def main():
    raw = sys.argv[1:]
    verbose = "-v" in raw or "--steps" in raw
    want_fraction = "-f" in raw or "--fraction" in raw
    args = [a for a in raw if a not in ("-v", "--steps", "-f", "--fraction")]

    if args:
        equation = args[0]
    else:
        try:
            equation = input()
        except EOFError:
            print("Error: no equation given.", file=sys.stderr)
            return 1

    try:
        run(equation, verbose, want_fraction)
    except ParseError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ZeroDivisionError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())