"""Runtime safety hooks for the FundWise Render process.

This module is imported automatically by Python when the repository root is on
sys.path. It keeps third-party historical-NAV calls from holding the CAS request
for the full five-second per-call timeout when the upstream service is slow.
"""
import logging

try:
    import httpx

    _logger = logging.getLogger("fundwise.runtime")
    _original_get = httpx.AsyncClient.get

    async def _fundwise_bounded_get(self, url, *args, **kwargs):
        # FundWise only uses AsyncClient.get for the AMFI/MFAPI historical NAV
        # lookup. Keep that dependency bounded so a slow upstream cannot make
        # a CAS analysis appear to hang indefinitely.
        if isinstance(url, str) and "api.mfapi.in/mf/" in url:
            kwargs["timeout"] = min(float(kwargs.get("timeout", 5.0)), 2.0)
        return await _original_get(self, url, *args, **kwargs)

    httpx.AsyncClient.get = _fundwise_bounded_get
    _logger.info("FundWise runtime safety hooks loaded")
except Exception:
    # Never prevent the application from starting if the optional hook fails.
    pass
