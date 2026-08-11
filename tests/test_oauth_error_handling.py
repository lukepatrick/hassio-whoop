#!/usr/bin/env python3
"""Guards for OAuth 401 recovery and error classification.

Background
----------
WHOOP has been observed returning 401 for an access token that Home Assistant
still considered valid (`expires_at` in the future). Five such outages were
recorded 2026-07-30 .. 08-10, lasting 6-27 hours each.

They lasted hours because of how the failure was classified. Core only re-arms
the coordinator's poll timer `if not auth_failed`
(helpers/update_coordinator.py), so raising ConfigEntryAuthFailed stops polling
*permanently* - the integration cannot recover on its own and waits for a human.
Yet two of those five outages were healed with no browser consent at all (one by
an HA restart, one by `reload_config_entry`), proving the refresh grant was alive
the whole time and a refresh was all that was needed.

So the integration now forces one token refresh, retries for AUTH_RETRY_WINDOW,
and only then escalates to reauth.

These are structural assertions, parsed from the real source with `ast`. They
need no Home Assistant, no pytest and no stubbing, and unlike a behavioural test
against fakes they cannot pass vacuously.

Run with plain CPython, no dependencies::

    python3 tests/test_oauth_error_handling.py
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INIT_PY = REPO_ROOT / "custom_components" / "whoop" / "__init__.py"
API_PY = REPO_ROOT / "custom_components" / "whoop" / "api.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find(tree: ast.AST, name: str, kinds: tuple) -> ast.AST | None:
    for node in ast.walk(tree):
        if isinstance(node, kinds) and getattr(node, "name", None) == name:
            return node
    return None


def _func(tree: ast.AST, name: str):
    return _find(tree, name, (ast.AsyncFunctionDef, ast.FunctionDef))


def _base_names(cls: ast.ClassDef) -> set[str]:
    out = set()
    for base in cls.bases:
        if isinstance(base, ast.Attribute):
            out.add(base.attr)
        elif isinstance(base, ast.Name):
            out.add(base.id)
    return out


def _called_names(node: ast.AST) -> set[str]:
    """Names invoked as calls beneath `node`, both `foo()` and `x.foo()`."""
    out = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Attribute):
                out.add(child.func.attr)
            elif isinstance(child.func, ast.Name):
                out.add(child.func.id)
    return out


def _raised_names(node: ast.AST) -> set[str]:
    out = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is not None:
            exc = child.exc
            if isinstance(exc, ast.Call):
                exc = exc.func
            if isinstance(exc, ast.Attribute):
                out.add(exc.attr)
            elif isinstance(exc, ast.Name):
                out.add(exc.id)
    return out


def _handler_order(func: ast.AST) -> list[set[str]]:
    """Except-handler type names, in source order, for the Try blocks in func."""
    order: list[set[str]] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                names: set[str] = set()
                targets = []
                if isinstance(handler.type, ast.Tuple):
                    targets = list(handler.type.elts)
                elif handler.type is not None:
                    targets = [handler.type]
                for t in targets:
                    if isinstance(t, ast.Attribute):
                        names.add(t.attr)
                    elif isinstance(t, ast.Name):
                        names.add(t.id)
                order.append(names)
    return order


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_unauthorized_is_neither_update_failed_nor_auth_failed() -> None:
    """The single most breakable invariant in this design.

    Subclass UpdateFailed -> the per-endpoint `except UpdateFailed: return None`
    guards in api.py swallow it, async_update_data never sees the 401, and the
    integration silently serves empty data forever.

    Subclass ConfigEntryAuthFailed -> core stops the coordinator permanently on
    the first 401 and we are back to multi-hour outages needing a human.
    """
    cls = _find(_parse(API_PY), "WhoopUnauthorized", (ast.ClassDef,))
    assert cls is not None, "WhoopUnauthorized not found in api.py"

    bases = _base_names(cls)
    assert "UpdateFailed" not in bases, (
        "WhoopUnauthorized must NOT subclass UpdateFailed - the get_* helpers "
        "swallow UpdateFailed and return None, so the 401 would never reach "
        "async_update_data and the integration would serve empty data silently."
    )
    assert "ConfigEntryAuthFailed" not in bases, (
        "WhoopUnauthorized must NOT subclass ConfigEntryAuthFailed - core stops "
        "the coordinator permanently on auth failure, so a recoverable 401 "
        "would strand the integration until a human intervened."
    )


def test_api_maps_401_to_unauthorized_not_auth_failed() -> None:
    """api.py classifies; it must not escalate."""
    tree = _parse(API_PY)
    raised = _raised_names(tree)

    assert "WhoopUnauthorized" in raised, "api.py no longer raises WhoopUnauthorized"
    assert "ConfigEntryAuthFailed" not in raised, (
        "api.py raises ConfigEntryAuthFailed again. That is terminal for the "
        "coordinator; only async_update_data may decide a 401 is fatal."
    )


def test_api_does_not_force_token_refresh() -> None:
    """The retry must live in async_update_data, never in api.py.

    async_update_data fires six requests through asyncio.gather, so a dead token
    yields six simultaneous 401s. Forcing a refresh per-request would present
    the same rotating refresh token six times concurrently - textbook
    reuse-detection input, which could turn a recoverable fault into a genuinely
    revoked grant.
    """
    called = _called_names(_parse(API_PY))
    for forbidden in ("async_update_entry", "_invalidate_access_token",
                      "async_ensure_token_valid"):
        assert forbidden not in called, (
            f"api.py calls {forbidden}(). Token refresh must happen once per "
            "update cycle in async_update_data, not per-request - six "
            "concurrent refreshes risk WHOOP revoking the whole grant."
        )


def test_invalidate_rebuilds_the_token_dict() -> None:
    """Mutating the nested token dict in place silently persists nothing.

    async_update_entry() bails out on `entry.data != data`; an in-place edit
    compares equal, so the change is never saved and the next cycle would not
    refresh.
    """
    func = _func(_parse(INIT_PY), "_invalidate_access_token")
    assert func is not None, "_invalidate_access_token not found in __init__.py"

    assert "async_update_entry" in _called_names(func), (
        "_invalidate_access_token must persist via async_update_entry()"
    )
    # A rebuild looks like {**token, "expires_at": 0}; an in-place mutation
    # looks like token["expires_at"] = 0.
    has_dict_rebuild = any(isinstance(n, ast.Dict) for n in ast.walk(func))
    mutates_subscript = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Subscript) for t in n.targets)
        for n in ast.walk(func)
    )
    assert has_dict_rebuild, "expected a rebuilt dict literal, e.g. {**token, ...}"
    assert not mutates_subscript, (
        "_invalidate_access_token mutates the token dict by subscript. "
        "async_update_entry compares entry.data != data, so an in-place edit "
        "compares equal and is silently never persisted."
    )


def test_oauth_token_request_error_reaches_core() -> None:
    """D1: the blanket `except Exception` must not swallow OAuth token errors.

    OAuth2TokenRequestError subclasses ClientResponseError, not IntegrationError,
    so it is neither ConfigEntryAuthFailed nor UpdateFailed. If the blanket
    handler catches it, a genuinely revoked grant becomes an UpdateFailed - the
    coordinator retries forever and NEVER prompts for reauth.
    """
    func = _func(_parse(INIT_PY), "async_update_data")
    assert func is not None, "async_update_data not found"

    order = _handler_order(func)
    idx_oauth = next(
        (i for i, names in enumerate(order) if "OAuth2TokenRequestError" in names), None
    )
    idx_blanket = next(
        (i for i, names in enumerate(order) if "Exception" in names), None
    )

    assert idx_oauth is not None, (
        "async_update_data no longer re-raises OAuth2TokenRequestError. The "
        "blanket handler will swallow it into UpdateFailed and a revoked grant "
        "will spin forever without ever prompting the user."
    )
    if idx_blanket is not None:
        assert idx_oauth < idx_blanket, (
            "OAuth2TokenRequestError must be caught BEFORE `except Exception`, "
            f"got positions {idx_oauth} and {idx_blanket}."
        )


def test_transient_setup_failures_are_not_auth_failures() -> None:
    """D2: a network blip must not demand browser consent.

    _token_request only wraps ClientResponseError, so raw network errors arrive
    unwrapped - hence ClientError/TimeoutError must be caught explicitly.
    ConfigEntryNotReady retries with backoff; ConfigEntryAuthFailed nags the user.
    """
    func = _func(_parse(INIT_PY), "async_setup_entry")
    assert func is not None, "async_setup_entry not found"

    order = _handler_order(func)
    transient = next(
        (i for i, n in enumerate(order) if "OAuth2TokenRequestError" in n), None
    )
    assert transient is not None, (
        "async_setup_entry no longer distinguishes transient token-endpoint "
        "failures; a WHOOP 5xx during startup will demand a browser reauth."
    )
    assert any("ClientError" in n for n in order), (
        "async_setup_entry must also catch raw ClientError/TimeoutError - "
        "_token_request only wraps ClientResponseError, so network failures "
        "arrive unwrapped and would hit the blanket ConfigEntryAuthFailed."
    )
    assert "ConfigEntryNotReady" in _raised_names(func), (
        "async_setup_entry must raise ConfigEntryNotReady for transient "
        "failures so HA retries with backoff instead of prompting the user."
    )

    reauth = next(
        (i for i, n in enumerate(order) if "OAuth2TokenRequestReauthError" in n), None
    )
    assert reauth is not None and reauth < transient, (
        "OAuth2TokenRequestReauthError is a subclass of OAuth2TokenRequestError "
        "and must be caught first, or a dead grant is treated as transient and "
        "the user is never prompted."
    )


def main() -> int:
    tests = [
        test_unauthorized_is_neither_update_failed_nor_auth_failed,
        test_api_maps_401_to_unauthorized_not_auth_failed,
        test_api_does_not_force_token_refresh,
        test_invalidate_rebuilds_the_token_dict,
        test_oauth_token_request_error_reaches_core,
        test_transient_setup_failures_are_not_auth_failures,
    ]

    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as err:
            failures += 1
            print(f"FAIL  {test.__name__}\n      {err}")
        except Exception as err:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}\n      {type(err).__name__}: {err}")
        else:
            print(f"ok    {test.__name__}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
