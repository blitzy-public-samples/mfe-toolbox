"""
Pytest tests for the SARIMA empty stub function.

This module tests that ``mfe_toolbox.sandbox.sarima.sarima`` — migrated from
the MATLAB ``sandbox/sarima.m`` empty function — correctly raises
``NotImplementedError`` for all invocations, preserves the original 11-parameter
function signature using snake_case Python naming, and carries proper docstrings.

The original MATLAB file (Version 4.0, Kevin Sheppard, University of Oxford)
contained only a function declaration line with no body:

.. code-block:: matlab

    function sarima(y,c,p,q,d,seasonal,x,startingVals,options,holdBack,sigma2)

Per AAP Section 0.7.2 the Python equivalent must be a stub that unconditionally
raises ``NotImplementedError``.  There are **no** fixture-based parity tests
because the MATLAB function performed no computation.

References
----------
Migrated from ``sandbox/sarima.m`` in the MFE Toolbox (Version 4.0).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from mfe_toolbox.sandbox.sarima import sarima


# ---------------------------------------------------------------------------
# Phase 1: Import and Stub Behavior Tests
# ---------------------------------------------------------------------------


class TestSarimaImportAndCallable:
    """Verify that the sarima function can be imported and is callable."""

    def test_sarima_imports(self) -> None:
        """Importing sarima from mfe_toolbox.sandbox.sarima must succeed and
        the imported name must be a callable object."""
        # Re-import inside the test body to exercise the import path explicitly
        from mfe_toolbox.sandbox.sarima import sarima as _sarima  # noqa: F811

        assert _sarima is not None, "sarima import returned None"
        assert callable(_sarima), "sarima should be callable"

    def test_sarima_is_callable(self) -> None:
        """``sarima`` must be a callable (function) object."""
        assert callable(sarima), (
            "Expected sarima to be callable, got type "
            f"{type(sarima).__name__}"
        )


class TestSarimaRaisesNotImplemented:
    """Core tests verifying that sarima always raises NotImplementedError."""

    def test_sarima_raises_not_implemented(self) -> None:
        """Calling sarima with minimal positional args (a small array) must
        raise ``NotImplementedError``.  This is the CRITICAL stub-behaviour
        test required by AAP Section 0.7.2."""
        with pytest.raises(NotImplementedError):
            sarima(np.array([1.0, 2.0, 3.0]))

    def test_sarima_raises_with_all_args(self) -> None:
        """Calling sarima with every keyword argument populated must still
        raise ``NotImplementedError``."""
        with pytest.raises(NotImplementedError):
            sarima(
                y=np.array([1.0]),
                c=1,
                p=1,
                q=1,
                d=1,
                seasonal=np.array([[1, 1, 12]]),
                x=np.array([[1.0]]),
                starting_vals=np.array([0.5]),
                options={},
                hold_back=10,
                sigma2=np.array([1.0]),
            )

    def test_sarima_raises_with_no_args(self) -> None:
        """Calling sarima with **no** arguments must never return successfully.

        Because the Python implementation defaults every parameter to ``None``,
        the function should still reach the ``raise NotImplementedError``
        statement (rather than raising ``TypeError`` for a missing positional
        argument)."""
        with pytest.raises(NotImplementedError):
            sarima()

    def test_sarima_error_message(self) -> None:
        """The ``NotImplementedError`` message must indicate that the function
        is not implemented (case-insensitive).  The actual message may read
        'not yet implemented' or 'not implemented'."""
        with pytest.raises(NotImplementedError) as exc_info:
            sarima(np.array([1.0]))

        message = str(exc_info.value).lower()
        # Accept both "not implemented" and "not yet implemented" phrasings
        assert "implemented" in message and "not" in message, (
            f"Error message '{exc_info.value}' does not convey "
            "a 'not implemented' intent"
        )


# ---------------------------------------------------------------------------
# Phase 2: Function Signature Tests
# ---------------------------------------------------------------------------


class TestSarimaSignature:
    """Verify that the function signature preserves the MATLAB parameter names
    using snake_case Python conventions."""

    # Expected parameter names after MATLAB → Python snake_case conversion:
    #   startingVals → starting_vals
    #   holdBack     → hold_back
    EXPECTED_PARAMS: tuple[str, ...] = (
        "y",
        "c",
        "p",
        "q",
        "d",
        "seasonal",
        "x",
        "starting_vals",
        "options",
        "hold_back",
        "sigma2",
    )

    def test_sarima_preserves_function_signature(self) -> None:
        """``inspect.signature(sarima)`` must expose exactly 11 parameters
        whose names match the MATLAB originals translated to snake_case."""
        sig = inspect.signature(sarima)
        param_names = tuple(sig.parameters.keys())

        assert param_names == self.EXPECTED_PARAMS, (
            f"Signature mismatch.\n"
            f"  Expected: {self.EXPECTED_PARAMS}\n"
            f"  Got:      {param_names}"
        )

    def test_sarima_parameter_count(self) -> None:
        """The function must accept exactly 11 parameters, matching the
        original MATLAB ``sarima.m`` declaration."""
        sig = inspect.signature(sarima)
        assert len(sig.parameters) == 11, (
            f"Expected 11 parameters, found {len(sig.parameters)}"
        )

    def test_sarima_all_params_have_defaults(self) -> None:
        """Every parameter should have a default value (``None``), reflecting
        the MATLAB ``nargin``-based optional-argument convention."""
        sig = inspect.signature(sarima)
        for name, param in sig.parameters.items():
            assert param.default is not inspect.Parameter.empty, (
                f"Parameter '{name}' has no default value; expected None"
            )


# ---------------------------------------------------------------------------
# Phase 3: Module-Level / Docstring Tests
# ---------------------------------------------------------------------------


class TestSarimaDocstrings:
    """Ensure the module and function carry appropriate documentation."""

    def test_sarima_module_has_docstring(self) -> None:
        """The ``mfe_toolbox.sandbox.sarima`` module must have a non-empty
        module-level docstring."""
        import mfe_toolbox.sandbox.sarima as sarima_mod

        assert sarima_mod.__doc__ is not None, (
            "Module mfe_toolbox.sandbox.sarima has no docstring"
        )
        assert len(sarima_mod.__doc__.strip()) > 0, (
            "Module docstring is empty"
        )

    def test_sarima_function_has_docstring(self) -> None:
        """The ``sarima`` function itself must have a non-empty docstring."""
        assert sarima.__doc__ is not None, (
            "Function sarima() has no docstring"
        )
        assert len(sarima.__doc__.strip()) > 0, (
            "Function docstring is empty"
        )

    def test_sarima_accessible_from_sandbox_package(self) -> None:
        """``sarima`` must be importable from the parent sandbox package,
        i.e. ``from mfe_toolbox.sandbox import sarima`` must succeed and
        the resulting name must be the module (the ``__init__.py`` exports
        it by module name, not function name)."""
        from mfe_toolbox.sandbox import sarima as sarima_ref  # noqa: F811

        # The import may resolve to the module or the function depending on
        # how __init__.py exports it.  Either way the name must be truthy.
        assert sarima_ref is not None, (
            "Could not import sarima from mfe_toolbox.sandbox"
        )
