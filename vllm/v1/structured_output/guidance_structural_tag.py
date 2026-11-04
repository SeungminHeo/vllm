# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lower xgrammar-style structural tags to llguidance Lark grammars.

The structural tag format is the one xgrammar defines
(``{"type": "structural_tag", "format": {...}}``) and that vLLM tool parsers
and the Dynamo frontend emit. llguidance has no reader for it, so each format
node is lowered to Lark here.

Two llguidance behaviours shape the lowering:

* Tokens llguidance treats as special (this includes added tokens the HF
  tokenizer marks ``special=False``, such as ``<tool_call>``) never match
  string literals or regexes. Literal text is split around the special-token
  strings of the tokenizer and those parts become ``<[id]>``.
* Lexemes are greedy. Free text followed by a plain-text terminator becomes
  one ``[lazy]`` rule that stops at the first terminator. Free text that may
  not contain a trigger excludes it with ``& ~``.
"""

import json
from collections.abc import Callable, Mapping
from typing import Any

_ANY = r"(?s:.*)"
UNSUPPORTED = "is not supported by the guidance structured outputs backend"


def special_token_strings(ll_tokenizer: Any) -> dict[str, int]:
    """Map the text of every token llguidance treats as special to its id."""
    out: dict[str, int] = {}
    for tok_id in range(ll_tokenizer.vocab_size):
        if ll_tokenizer.is_special_token(tok_id):
            text = ll_tokenizer.decode_bytes([tok_id]).decode("utf-8", "replace")
            if text:
                out.setdefault(text, tok_id)
    return out


def _regex(pattern: str) -> str:
    return "/" + pattern.replace("/", "\\/") + "/"


def _escape(text: str) -> str:
    out = []
    for ch in text:
        if ch in "\\.^$|?*+()[]{}-&~":
            out.append("\\" + ch)
        elif ch in "\n\t\r":
            out.append({"\n": "\\n", "\t": "\\t", "\r": "\\r"}[ch])
        else:
            out.append(ch)
    return "".join(out)


class _Lowering:
    def __init__(
        self, special_tokens: Mapping[str, int], process_schema: Callable[[Any], Any]
    ):
        # Longest first, so "<|im_end|>" wins over a token that is its prefix.
        self.special = sorted(special_tokens.items(), key=lambda kv: -len(kv[0]))
        self.process_schema = process_schema
        self.rules: list[str] = []
        self.terminals: dict[str, str] = {}
        self.side_grammars: list[dict[str, str]] = []
        self.n = 0

    def rule(self, body: str, lazy: bool = False) -> str:
        self.n += 1
        name = f"r{self.n}"
        self.rules.append(f"{name}{'[lazy]' if lazy else ''}: {body}")
        return name

    # ---- literals -----------------------------------------------------------
    def split(self, text: str) -> list[tuple[str, int | None]]:
        """Split text into plain pieces (id None) and special-token pieces."""
        out: list[tuple[str, int | None]] = []
        i = start = 0
        while i < len(text):
            hit = next(
                ((t, tid) for t, tid in self.special if text.startswith(t, i)), None
            )
            if hit is None:
                i += 1
                continue
            if start < i:
                out.append((text[start:i], None))
            out.append(hit)
            i += len(hit[0])
            start = i
        if start < len(text):
            out.append((text[start:], None))
        return out

    def atoms(self, text: str) -> list[str]:
        """Lark atoms matching `text`; a plain atom is a quoted string."""
        return [
            json.dumps(t) if tid is None else f"<[{tid}]>"
            for t, tid in self.split(text)
        ]

    def token(self, tok: int | str) -> str:
        if isinstance(tok, int):
            return f"<[{tok}]>"
        for text, tid in self.special:
            if text == tok:
                return f"<[{tid}]>"
        raise ValueError(f"token {tok!r} is not a special token of this tokenizer")

    def marker(self, value: Any) -> list[str]:
        """Atoms for a tag begin/end: a string, a list of strings or a token."""
        if isinstance(value, dict):
            if value.get("type") != "token":
                raise ValueError(f"tag begin/end {value!r} {UNSUPPORTED}")
            return [self.token(value["token"])]
        if isinstance(value, list):
            alts = [" ".join(self.atoms(v)) or self.rule("") for v in value]
            return ["(" + " | ".join(alts) + ")"]
        return self.atoms(value)

    # ---- free text --------------------------------------------------------------
    def text(self, excludes: list[str]) -> str:
        """Terminal for free text containing none of `excludes`.

        Excludes containing a special token need no handling, text can never
        produce one.
        """
        plain = sorted(
            {
                e
                for e in excludes
                if e and len(self.split(e)) == 1 and self.split(e)[0][1] is None
            }
        )
        regex = _regex(_ANY)
        if plain:
            alt = "|".join(_escape(e) for e in plain)
            regex += f" & ~{_regex(_ANY + '(?:' + alt + ')' + _ANY)}"
        if regex not in self.terminals:
            self.n += 1
            self.terminals[regex] = f"T{self.n}"
            self.rules.append(f"T{self.n}: {regex}")
        return self.terminals[regex]

    def text_then(self, excludes: list[str], after: list[str]) -> str:
        """Free text followed by `after`, ending the text at the first `after`."""
        text = self.text(excludes)
        if after and _is_lexeme(after[0]):
            # Plain-text terminator: one lazy lexeme ends at its first match.
            head = self.rule(f"{text} {after[0]}", lazy=True)
            return " ".join([head, *after[1:]])
        # Empty or starting with a special token, which free text cannot
        # consume, so greedy text stops in front of it.
        return " ".join([text, *after])

    # ---- formats ----------------------------------------------------------------
    def fmt(self, f: Mapping[str, Any]) -> str:
        handler = getattr(self, f"f_{f.get('type')}", None)
        if handler is None:
            raise ValueError(f"structural tag format {f.get('type')!r} {UNSUPPORTED}")
        return handler(f)

    def lead(self, f: Mapping[str, Any]) -> list[str]:
        """`f` as atoms, with its leading literal split out where it has one."""
        if f.get("type") == "const_string":
            return self.atoms(f["value"])
        if f.get("type") == "tag" and not isinstance(f["begin"], dict):
            return [*self.marker(f["begin"]), self.tag_body(f)]
        return [self.fmt(f)]

    def f_const_string(self, f):
        return " ".join(self.atoms(f["value"])) or self.rule("")

    def f_json_schema(self, f):
        if f.get("style", "json") != "json":
            raise ValueError(f"json_schema style {f['style']!r} {UNSUPPORTED}")
        if f.get("any_order"):
            raise ValueError(f"json_schema any_order {UNSUPPORTED}")
        schema = f["json_schema"]
        if schema is False:
            raise ValueError("json_schema false matches nothing")
        return "%json " + json.dumps(
            self.process_schema({} if schema is True else schema)
        )

    def f_regex(self, f):
        return self.rule(_regex(f["pattern"]))

    def f_grammar(self, f):
        import llguidance

        name = f"g{len(self.side_grammars) + 1}"
        self.side_grammars.append(
            {"name": name, "lark_grammar": llguidance.gbnf_to_lark(f["grammar"])}
        )
        return "@" + name

    def f_any_text(self, f):
        _free_text_bounds(f)
        return self.text(_excludes(f))

    f_any_tokens = f_any_text

    def f_sequence(self, f):
        elems, out, i = f["elements"], [], 0
        while i < len(elems):
            if _is_free_text(elems[i]) and i + 1 < len(elems):
                out.append(self.text_then(_excludes(elems[i]), self.lead(elems[i + 1])))
                i += 2
            else:
                out.append(self.fmt(elems[i]))
                i += 1
        return self.rule(" ".join(out))

    def f_or(self, f):
        return "(" + " | ".join(self.fmt(e) for e in f["elements"]) + ")"

    def f_optional(self, f):
        return f"({self.fmt(f['content'])})?"

    def f_star(self, f):
        return f"({self.fmt(f['content'])})*"

    def f_plus(self, f):
        return f"({self.fmt(f['content'])})+"

    def f_repeat(self, f):
        lo, hi = int(f["min"]), int(f["max"])
        body = self.rule(self.fmt(f["content"]))
        return f"{body}{{{lo},}}" if hi < 0 else f"{body}{{{lo},{hi}}}"

    def tag_body(self, f) -> str:
        """Everything of a tag after its begin marker."""
        end = self.marker(f["end"])
        if _is_free_text(f["content"]):
            return self.rule(self.text_then(_excludes(f["content"]), end))
        return self.rule(" ".join([self.fmt(f["content"]), *end]))

    def f_tag(self, f):
        return self.rule(" ".join([*self.marker(f["begin"]), self.tag_body(f)]))

    def f_triggered_tags(self, f):
        """Free text with tags. As in xgrammar, ``at_least_one`` makes the
        output start with a tag, and ``stop_after_first`` ends it after one."""
        triggers, excludes = f["triggers"], list(f.get("excludes", []))
        bare, after_text = [], []
        for t in f["tags"]:
            if not isinstance(t.get("begin"), str):
                raise ValueError(f"triggered_tags with a token begin {UNSUPPORTED}")
            trig = next((tr for tr in triggers if t["begin"].startswith(tr)), None)
            if trig is None:
                raise ValueError(f"tag begin {t['begin']!r} matches no trigger")
            body = self.rule(
                " ".join([*self.atoms(t["begin"][len(trig) :]), self.tag_body(t)])
            )
            bare.append(self.rule(" ".join([*self.atoms(trig), body])))
            after_text.append(
                self.rule(self.text_then(excludes, [*self.atoms(trig), body]))
            )
        first = "(" + " | ".join(bare) + ")"
        later = "(" + " | ".join(after_text) + ")"
        # Text after the last tag may not contain a trigger: a trigger always
        # starts a tag.
        tail = self.text(excludes + triggers)
        at_least_one = f.get("at_least_one", False)
        if f.get("stop_after_first", False):
            return first if at_least_one else f"({later} | {tail})"
        if at_least_one:
            return f"({first} {later}* {tail})"
        return f"({later}* {tail})"

    def f_tags_with_separator(self, f):
        one = "(" + " | ".join(self.f_tag(t) for t in f["tags"]) + ")"
        sep = " ".join(self.atoms(f.get("separator", "")))
        if f.get("stop_after_first", False):
            many = one
        else:
            many = f"{one} ({sep} {one})*" if sep else f"{one}+"
        return f"({many})" if f.get("at_least_one", False) else f"({many})?"


def _is_lexeme(atom: str) -> bool:
    return atom.startswith('"')


def _free_text_bounds(f: Mapping[str, Any]) -> None:
    if f.get("max_tokens") is not None or f.get("max_chars") is not None:
        raise ValueError(f"{f['type']} with max_tokens/max_chars {UNSUPPORTED}")
    if any(not isinstance(t, str) for t in f.get("exclude_tokens", [])):
        raise ValueError(f"{f['type']} with token ids {UNSUPPORTED}")


def _is_free_text(f: Mapping[str, Any]) -> bool:
    if f.get("type") not in ("any_text", "any_tokens"):
        return False
    _free_text_bounds(f)
    return True


def _excludes(f: Mapping[str, Any]) -> list[str]:
    # any_tokens is lowered at text level: its excluded tokens may not appear.
    return list(f.get("excludes", [])) + list(f.get("exclude_tokens", []))


def structural_tag_to_guidance_grammar(
    spec: Mapping[str, Any],
    special_tokens: Mapping[str, int],
    process_schema: Callable[[Any], Any],
) -> str:
    """Lower a structural tag to an llguidance grammar.

    Args:
        spec: Parsed ``{"type": "structural_tag", "format": {...}}``.
        special_tokens: Text to id of the tokens llguidance treats as special.
        process_schema: Applied to every JSON schema before it is inlined.

    Returns:
        A Lark grammar, or a JSON top-level grammar when the tag embeds
        ``grammar`` formats.

    Raises:
        ValueError: The tag uses a format the guidance backend cannot express.

    """
    if spec.get("type", "structural_tag") != "structural_tag" or "format" not in spec:
        raise ValueError("structural tag must have a 'format' field")
    lw = _Lowering(special_tokens, process_schema)
    start = lw.fmt(spec["format"])
    lark = "\n".join(["%llguidance {}", f"start: {start}", *lw.rules]) + "\n"
    if not lw.side_grammars:
        return lark
    grammars = [{"name": "structural_tag", "lark_grammar": lark}, *lw.side_grammars]
    return json.dumps({"grammars": grammars})
