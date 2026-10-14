"""Text folding shared by ingestion and querying.

PDF fonts emit typographic ligatures ("ﬁ", "ﬂ") and non-breaking spaces, which break keyword
search unless they are folded to plain letters and spaces. Unicode's compatibility form (NFKC)
does that, but it also rewrites characters whose compatibility form changes a *number*:
"10³" becomes "103", "1½" becomes "11⁄2", "cm²" becomes "cm2". In a compliance manual that is
a different limit, so those three classes are left exactly as the document has them.
"""

from __future__ import annotations

import unicodedata

#: Compatibility decompositions that would change the meaning of a quantity.
_LOSSY_PREFIXES = ("<super>", "<sub>", "<fraction>")
_CONTROL_KEEP = {"\t", "\n", "\r"}


def fold_compat(text: str) -> str:
    """NFC, then NFKC for every character except superscripts, subscripts and fractions."""
    text = unicodedata.normalize("NFC", text)
    return "".join(
        ch
        if unicodedata.decomposition(ch).startswith(_LOSSY_PREFIXES)
        else unicodedata.normalize("NFKC", ch)
        for ch in text
    )


def has_control_characters(text: str) -> bool:
    """True when ``text`` holds a C0/C1 control character other than tab or a line break."""
    return any(unicodedata.category(ch) == "Cc" and ch not in _CONTROL_KEEP for ch in text)
