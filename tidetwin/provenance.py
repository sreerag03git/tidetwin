"""Provenance-carrying quantities.

Every number this application displays must be traceable. A ``Quantity`` binds a
value to its units, its provenance class, its citation, and the chain of inputs
it was computed from. Arithmetic on ``Quantity`` objects produces ``DERIVED``
results that retain the full input DAG, so the provenance card for any displayed
figure can be reconstructed by walking ``Quantity.chain()``.

Provenance classes
------------------
MEASURED
    From a real external dataset (ERA5, TPXO, digitised published test data).
    Must cite dataset, variable and retrieval date.
PUBLISHED
    A constant taken from a standard or paper. Must cite document and locator.
DERIVED
    Computed by our own solvers from MEASURED and/or PUBLISHED inputs.
ASSUMED
    A value the user set, or a placeholder with no external authority. Renders
    red. Every downstream result inherits ``contaminated == True``.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

import numpy as np

__all__ = [
    "Provenance",
    "Citation",
    "Quantity",
    "measured",
    "published",
    "derived",
    "assumed",
    "unavailable",
    "DataUnavailable",
    "as_value",
]


class Provenance(Enum):
    """The four permitted provenance classes."""

    MEASURED = "MEASURED"
    PUBLISHED = "PUBLISHED"
    DERIVED = "DERIVED"
    ASSUMED = "ASSUMED"

    @property
    def colour(self) -> str:
        """Hex colour for the provenance chip.

        Chosen for a white background and checked for contrast: each of these
        clears WCAG AA (4.5:1) against #ffffff, so the chip is legible as text
        and not only as a colour. Colour is never the sole carrier of meaning -
        the class name is always spelled out beside it.
        """
        return {
            "MEASURED": "#1a7f43",  # green
            "PUBLISHED": "#1a5fb4",  # blue
            "DERIVED": "#4a545e",  # grey
            "ASSUMED": "#b3261e",  # red
        }[self.value]


class DataUnavailable(RuntimeError):
    """Raised when a required external data source is absent.

    Callers must surface this as an explicit ``DATA UNAVAILABLE`` state and mark
    the dependent claim ``UNTESTABLE - DATA MISSING``. Substituting a synthetic
    stand-in in place of the real source is forbidden.
    """

    def __init__(self, source: str, reason: str, remedy: str = "") -> None:
        self.source = source
        self.reason = reason
        self.remedy = remedy
        super().__init__(f"DATA UNAVAILABLE - {source}: {reason}")


@dataclass(frozen=True)
class Citation:
    """A bibliographic pointer precise enough to check."""

    document: str
    locator: str = ""
    year: int | None = None
    url: str | None = None
    retrieved: _dt.date | None = None
    variable: str | None = None

    def __str__(self) -> str:
        bits = [self.document]
        if self.year is not None:
            bits.append(f"({self.year})")
        if self.locator:
            bits.append(self.locator)
        if self.variable:
            bits.append(f"variable '{self.variable}'")
        if self.retrieved is not None:
            bits.append(f"retrieved {self.retrieved.isoformat()}")
        return ", ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "locator": self.locator,
            "year": self.year,
            "url": self.url,
            "retrieved": self.retrieved.isoformat() if self.retrieved else None,
            "variable": self.variable,
        }


def _is_array(x: Any) -> bool:
    return isinstance(x, np.ndarray)


@dataclass(frozen=True)
class Quantity:
    """A value that knows where it came from.

    Parameters
    ----------
    value
        Scalar or ``np.ndarray``. Arrays are supported so that time series and
        parameter sweeps carry provenance too.
    units
        SI-preferred unit string, e.g. ``"m"``, ``"Pa"``, ``"-"`` for
        dimensionless. Never empty: use ``"-"``.
    provenance
        One of :class:`Provenance`.
    name
        Short human label used in the provenance card and exports.
    citation
        Required for MEASURED and PUBLISHED. Optional otherwise.
    inputs
        Upstream quantities. Populated automatically by arithmetic and by
        :func:`derived`.
    uncertainty
        One standard deviation, in the same units as ``value``. ``None`` means
        not characterised (which is not the same as zero).
    operation
        Free-text description of the transform that produced a DERIVED value.
    note
        Caveats that belong next to the number.
    """

    value: float | np.ndarray
    units: str
    provenance: Provenance
    name: str = ""
    citation: Citation | None = None
    inputs: tuple["Quantity", ...] = field(default=())
    uncertainty: float | np.ndarray | None = None
    operation: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.units:
            raise ValueError(f"Quantity '{self.name}' has empty units; use '-' for dimensionless")
        if self.provenance in (Provenance.MEASURED, Provenance.PUBLISHED) and self.citation is None:
            raise ValueError(
                f"Quantity '{self.name}' is {self.provenance.value} but carries no citation. "
                "MEASURED and PUBLISHED values must be traceable."
            )
        # A Quantity is a number that knows where it came from. A string is not
        # one, and until this check existed it was accepted here and only failed
        # much later inside format(), where the traceback points at the renderer
        # rather than at the call that made the mistake.
        if isinstance(self.value, (str, bytes)):
            raise TypeError(
                f"Quantity '{self.name}' was given the {type(self.value).__name__} "
                f"{self.value!r}. Quantities are numeric - a yes/no or a label is not a "
                "measurement and cannot carry uncertainty or units. Write it as text, or "
                "give the number behind it."
            )

    # ------------------------------------------------------------------ chain

    def chain(self) -> list["Quantity"]:
        """Flattened, de-duplicated list of every quantity in this value's DAG.

        Ordered leaves-first so the reader sees the sources before the result.
        """
        seen: dict[int, Quantity] = {}
        order: list[Quantity] = []

        def walk(q: "Quantity") -> None:
            if id(q) in seen:
                return
            seen[id(q)] = q
            for parent in q.inputs:
                walk(parent)
            order.append(q)

        walk(self)
        return order

    @property
    def contaminated(self) -> bool:
        """True if any ancestor is ASSUMED.

        A contaminated result is not wrong, but its uncertainty is not bounded by
        anything external. The UI flags it and the claims ledger records it under
        ``blocking_assumptions``.
        """
        return any(q.provenance is Provenance.ASSUMED for q in self.chain())

    @property
    def blocking_assumptions(self) -> list[str]:
        """Names of the ASSUMED leaves this value rests on."""
        return sorted(
            {q.name or "<unnamed>" for q in self.chain() if q.provenance is Provenance.ASSUMED}
        )

    def sources(self) -> list["Quantity"]:
        """The MEASURED and PUBLISHED leaves underpinning this value."""
        return [
            q
            for q in self.chain()
            if q.provenance in (Provenance.MEASURED, Provenance.PUBLISHED)
        ]

    # ------------------------------------------------------------- formatting

    @property
    def is_array(self) -> bool:
        return _is_array(self.value)

    def format(self, sig: int = 4) -> str:
        """Value and units as a display string, with +/- uncertainty if known."""
        if self.is_array:
            arr = np.asarray(self.value)
            return f"array{arr.shape} {self.units}"
        v = float(self.value)
        if not math.isfinite(v):
            return f"{v} {self.units}"
        mag = abs(v)
        # Exact whole numbers are counts (joints, members, modes); rendering them
        # with a fixed number of significant figures reads as false precision.
        if v == int(v) and mag < 1e9 and self.uncertainty is None:
            unit = "" if self.units == "-" else f" {self.units}"
            return f"{int(v)}{unit}"
        if mag != 0 and (mag < 1e-3 or mag >= 1e5):
            txt = f"{v:.{sig}e}"
        else:
            decimals = max(0, sig - 1 - int(math.floor(math.log10(mag))) if mag > 0 else sig)
            txt = f"{v:.{min(decimals, 8)}f}"
        if self.uncertainty is not None and not _is_array(self.uncertainty):
            u = float(self.uncertainty)
            if math.isfinite(u) and u > 0:
                txt = f"{txt} +/- {u:.{max(1, sig - 1)}g}"
        unit = "" if self.units == "-" else f" {self.units}"
        return f"{txt}{unit}"

    def __str__(self) -> str:
        return f"{self.name or 'quantity'} = {self.format()} [{self.provenance.value}]"

    def to_dict(self, include_chain: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "value": (np.asarray(self.value).tolist() if self.is_array else float(self.value)),
            "units": self.units,
            "provenance": self.provenance.value,
            "uncertainty": (
                None
                if self.uncertainty is None
                else (
                    np.asarray(self.uncertainty).tolist()
                    if _is_array(self.uncertainty)
                    else float(self.uncertainty)
                )
            ),
            "citation": self.citation.to_dict() if self.citation else None,
            "operation": self.operation,
            "note": self.note,
            "contaminated": self.contaminated,
        }
        if include_chain:
            d["chain"] = [q.to_dict() for q in self.chain()[:-1]]
        return d

    # -------------------------------------------------------------- combinator

    def relabel(self, name: str, note: str = "") -> "Quantity":
        return replace(self, name=name, note=note or self.note)

    def with_note(self, note: str) -> "Quantity":
        joined = f"{self.note} {note}".strip() if self.note else note
        return replace(self, note=joined)

    def to(self, factor: float, units: str, name: str = "") -> "Quantity":
        """Unit conversion. ``factor`` multiplies the current value."""
        return Quantity(
            value=np.asarray(self.value) * factor if self.is_array else self.value * factor,
            units=units,
            provenance=Provenance.DERIVED if self.provenance is not Provenance.ASSUMED else Provenance.ASSUMED,
            name=name or self.name,
            inputs=(self,),
            uncertainty=None if self.uncertainty is None else np.asarray(self.uncertainty) * abs(factor),
            operation=f"unit conversion x{factor:g} -> {units}",
        )

    # -------------------------------------------------------------- arithmetic
    # First-order uncertainty propagation assuming independent inputs. Where
    # inputs are correlated this understates the result uncertainty; Monte Carlo
    # is used instead wherever correlation matters (see claims/tests/c3_*).

    def _binary(
        self,
        other: "Quantity | float | int | np.ndarray",
        fn: Callable[[Any, Any], Any],
        symbol: str,
        units: str,
        dfn: Callable[[Any, Any], tuple[Any, Any]] | None = None,
    ) -> "Quantity":
        if isinstance(other, Quantity):
            ov, ou, oname = other.value, other.uncertainty, other.name or "?"
            inputs: tuple[Quantity, ...] = (self, other)
        else:
            ov, ou, oname = other, None, _fmt_scalar(other)
            inputs = (self,)
        val = fn(self.value, ov)
        unc: Any = None
        if dfn is not None and (self.uncertainty is not None or ou is not None):
            da, db = dfn(self.value, ov)
            terms = 0.0
            if self.uncertainty is not None:
                terms = terms + (np.asarray(da) * np.asarray(self.uncertainty)) ** 2
            if ou is not None:
                terms = terms + (np.asarray(db) * np.asarray(ou)) ** 2
            unc = np.sqrt(terms)
            if not _is_array(val):
                unc = float(unc)
        prov = (
            Provenance.ASSUMED
            if any(q.provenance is Provenance.ASSUMED for q in inputs) and len(inputs) == 1
            else Provenance.DERIVED
        )
        return Quantity(
            value=val,
            units=units,
            provenance=prov,
            name=f"({self.name or '?'} {symbol} {oname})",
            inputs=inputs,
            uncertainty=unc,
            operation=f"{self.name or '?'} {symbol} {oname}",
        )

    def __add__(self, other):  # type: ignore[no-untyped-def]
        _check_units_match(self, other, "+")
        return self._binary(other, lambda a, b: a + b, "+", self.units, lambda a, b: (1.0, 1.0))

    def __sub__(self, other):  # type: ignore[no-untyped-def]
        _check_units_match(self, other, "-")
        return self._binary(other, lambda a, b: a - b, "-", self.units, lambda a, b: (1.0, -1.0))

    def __mul__(self, other):  # type: ignore[no-untyped-def]
        units = _mul_units(self.units, other.units if isinstance(other, Quantity) else "-")
        return self._binary(other, lambda a, b: a * b, "*", units, lambda a, b: (b, a))

    def __truediv__(self, other):  # type: ignore[no-untyped-def]
        units = _div_units(self.units, other.units if isinstance(other, Quantity) else "-")
        return self._binary(
            other, lambda a, b: a / b, "/", units, lambda a, b: (1.0 / b, -a / (b * b))
        )

    def __radd__(self, other):  # type: ignore[no-untyped-def]
        return self.__add__(other)

    def __rmul__(self, other):  # type: ignore[no-untyped-def]
        return self.__mul__(other)

    def __rsub__(self, other):  # type: ignore[no-untyped-def]
        return (self * -1.0).__add__(other)

    def __rtruediv__(self, other):  # type: ignore[no-untyped-def]
        num = assumed(other, self.units if False else "-", "scalar") if not isinstance(other, Quantity) else other
        return num.__truediv__(self)

    def __neg__(self) -> "Quantity":
        return self * -1.0

    def __pow__(self, p: float) -> "Quantity":
        return Quantity(
            value=self.value**p,
            units="-" if self.units == "-" else f"({self.units})^{p:g}",
            provenance=Provenance.DERIVED,
            name=f"({self.name or '?'})^{p:g}",
            inputs=(self,),
            uncertainty=(
                None
                if self.uncertainty is None
                else abs(p * np.asarray(self.value) ** (p - 1)) * np.asarray(self.uncertainty)
            ),
            operation=f"power {p:g}",
        )

    def __float__(self) -> float:
        if self.is_array:
            raise TypeError(f"Quantity '{self.name}' holds an array; use .value")
        return float(self.value)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        arr = np.asarray(self.value)
        return arr.astype(dtype) if dtype is not None else arr


def _fmt_scalar(x: Any) -> str:
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return "array"


def _check_units_match(a: Quantity, b: Any, op: str) -> None:
    if isinstance(b, Quantity) and a.units != b.units:
        raise ValueError(
            f"unit mismatch in '{a.name} {op} {b.name}': '{a.units}' vs '{b.units}'"
        )


def _mul_units(a: str, b: str) -> str:
    if a == "-":
        return b
    if b == "-":
        return a
    return f"{a}.{b}"


def _div_units(a: str, b: str) -> str:
    if b == "-":
        return a
    if a == b:
        return "-"
    return f"{a}/{b}"


# ------------------------------------------------------------------ factories


def measured(
    value: float | np.ndarray,
    units: str,
    name: str,
    citation: Citation,
    uncertainty: float | np.ndarray | None = None,
    note: str = "",
) -> Quantity:
    """A value read from a real external dataset."""
    return Quantity(value, units, Provenance.MEASURED, name, citation, (), uncertainty, "", note)


def published(
    value: float | np.ndarray,
    units: str,
    name: str,
    citation: Citation,
    uncertainty: float | np.ndarray | None = None,
    note: str = "",
) -> Quantity:
    """A constant from a standard or paper."""
    return Quantity(value, units, Provenance.PUBLISHED, name, citation, (), uncertainty, "", note)


def derived(
    value: float | np.ndarray,
    units: str,
    name: str,
    inputs: Sequence[Quantity],
    operation: str,
    uncertainty: float | np.ndarray | None = None,
    note: str = "",
    citation: Citation | None = None,
) -> Quantity:
    """A value computed by one of our solvers.

    ``operation`` should name the method precisely enough to find it in the
    source, e.g. ``"3D Timoshenko frame solve, K-joint chord saddle"``.
    """
    return Quantity(
        value,
        units,
        Provenance.DERIVED,
        name,
        citation,
        tuple(inputs),
        uncertainty,
        operation,
        note,
    )


def assumed(
    value: float | np.ndarray,
    units: str,
    name: str,
    note: str = "",
    uncertainty: float | np.ndarray | None = None,
) -> Quantity:
    """A user-set or placeholder value with no external authority.

    Renders red. Contaminates every downstream result.
    """
    return Quantity(value, units, Provenance.ASSUMED, name, None, (), uncertainty, "", note)


def unavailable(source: str, reason: str, remedy: str = "") -> DataUnavailable:
    """Construct (do not raise) the standard data-unavailable error."""
    return DataUnavailable(source, reason, remedy)


def as_value(x: Quantity | float | np.ndarray) -> Any:
    """Unwrap a ``Quantity`` for use inside a numeric kernel."""
    return x.value if isinstance(x, Quantity) else x


def combine(
    value: float | np.ndarray,
    units: str,
    name: str,
    inputs: Iterable[Quantity],
    operation: str,
    **kw: Any,
) -> Quantity:
    """Alias for :func:`derived` accepting any iterable of inputs."""
    return derived(value, units, name, list(inputs), operation, **kw)
