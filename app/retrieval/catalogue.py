"""The set of product names that actually exist in the corpus.

The router extracts a product name from the customer's own words - "the family
health policy", "my motor policy". Feeding that string straight into an exact
`MatchValue` filter is the problem this module exists to solve: the payload holds
the catalogue name ("Family Health Optima"), the customer's phrasing never equals
it, the filter matches zero points, and retrieval scores 0.0. The confidence gate
then fires and the whole search is re-run broadened - so every product-specific
question paid for two retrievals and used the filter from neither.

So a product name is only used as a filter once it has been *resolved* to a name
the corpus really contains. An unresolvable guess is dropped rather than applied,
because a filter that matches nothing is strictly worse than no filter at all.
"""

from __future__ import annotations

import re
import time

from sqlalchemy import select

from app.db.models import Document
from app.db.session import async_session_scope
from app.logging import get_logger

log = get_logger(__name__)

# Words that carry no product identity. "family health policy" and "family health
# plan" must both resolve to "Family Health Optima", so these are ignored when
# comparing. Without this, "policy" would have to appear in the catalogue name.
GENERIC = frozenset(
    {
        "policy",
        "policies",
        "plan",
        "insurance",
        "cover",
        "coverage",
        "product",
        "the",
        "my",
        "our",
        "a",
        "an",
    }
)

_CACHE: dict[str, tuple[float, tuple[str, ...]]] = {}
CACHE_TTL_SECONDS = 300.0


def _tokens(value: str) -> frozenset[str]:
    return frozenset(t for t in re.split(r"[^a-z0-9]+", value.lower()) if t and t not in GENERIC)


async def known_products(tenant_id: str) -> tuple[str, ...]:
    """Distinct non-null product names for a tenant, cached briefly.

    Cached because this runs on every retrieval and the catalogue changes only
    when a document is ingested. A short TTL is enough: a newly uploaded product
    becomes filterable within five minutes, and until then it simply is not used
    as a filter, which costs recall nothing.
    """
    now = time.monotonic()
    if (entry := _CACHE.get(tenant_id)) and now - entry[0] < CACHE_TTL_SECONDS:
        return entry[1]

    try:
        async with async_session_scope() as session:
            rows = await session.scalars(
                select(Document.product_name)
                .where(Document.tenant_id == tenant_id, Document.product_name.is_not(None))
                .distinct()
            )
            products = tuple(sorted({r for r in rows.all() if r and r.strip()}))
    except Exception as exc:
        # Never fail a search because the catalogue could not be read. No
        # catalogue means no product filter, which is the safe direction.
        log.warning("Could not load product catalogue", error=str(exc))
        return ()

    _CACHE[tenant_id] = (now, products)
    return products


def resolve(candidate: str | None, products: tuple[str, ...]) -> str | None:
    """Map the router's phrasing onto a catalogue name, or give up.

    Returns None when the guess cannot be resolved confidently, which means "do
    not filter on product". Giving up is the correct outcome far more often than
    guessing: the reranker still sees the whole tenant's corpus and sorts it by
    relevance, whereas a wrong product filter hides the only document that could
    have answered the question.

    A candidate resolves when every identifying token it carries appears in
    exactly one catalogue name. "family health policy" -> {family, health}, which
    is contained in "Family Health Optima" and nothing else. "motor policy" ->
    {motor} -> "Motor Shield Private Car". An ambiguous guess matching two
    products resolves to neither.
    """
    if not candidate or not products:
        return None

    wanted = _tokens(candidate)
    if not wanted:
        return None

    matches = [p for p in products if wanted <= _tokens(p)]
    if len(matches) == 1:
        return matches[0]

    if not matches:
        # Fall back to the reverse containment: the customer named the product
        # more fully than the catalogue does ("family health optima plan 2026"
        # against a stored "Family Health Optima").
        reverse = [p for p in products if _tokens(p) and _tokens(p) <= wanted]
        if len(reverse) == 1:
            return reverse[0]

    return None


def clear_cache() -> None:
    """Drop the cached catalogue. Called after an ingest so a newly uploaded
    product is filterable immediately rather than after the TTL."""
    _CACHE.clear()
