#!/usr/bin/env python3
"""
Enrich flags.json with basic country facts from Wikidata (CC0).

This script reads an existing `flags.json` produced by `download_flagcdn_flags.py`
and adds:
  - `wikidata`: { "qid": "Q...", "url": "https://www.wikidata.org/wiki/Q..." }
  - `facts`: up to N short strings (default: 3)

Notes:
  - Only ISO 3166-1 alpha-2 codes are supported (2-letter codes). Other codes
    (e.g. subdivisions like `gb-eng`) are left untouched.
  - Facts are derived from structured properties like capital, continent,
    languages, currency, population, and area.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

FACT_KEYS_DEFAULT = ["capital", "continent", "currency", "languages", "population", "area"]

ALBERTA_AREA_KM2 = 661_848.0

RESTCOUNTRIES_ALL_FIELDS_URL = "https://restcountries.com/v3.1/all?fields=cca2,area"

DEFAULT_AREA_ENTITY_OVERRIDES = {
    # Prefer "metropolitan France" (Europe + Corsica) over total France area which includes overseas.
    # Wikidata: https://www.wikidata.org/wiki/Q212429
    "fr": "Q212429",
}


@dataclass(frozen=True)
class CountryInfo:
    code: str  # lowercase alpha-2
    qid: str
    label: Optional[str]
    capital: Optional[str]
    continent: Optional[str]
    currencies: Optional[str]
    languages: Optional[str]
    population: Optional[int]
    area_km2: Optional[float]


def _http_get_json(url: str, timeout_s: int = 60) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "InkyPi (flags wikidata enrich; +https://github.com/mcdarby/InkyPi)",
            "Accept": "application/sparql-results+json",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def _qid_from_entity_uri(uri: str) -> str:
    # e.g. http://www.wikidata.org/entity/Q954
    return uri.rstrip("/").split("/")[-1]


def _format_number(n: float) -> str:
    return f"{n:,.0f}"


def _format_population(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _area_percent_of_alberta(area_km2: float) -> float:
    return (area_km2 / ALBERTA_AREA_KM2) * 100.0


def _format_area_vs_alberta(area_km2: float) -> str:
    percent = _area_percent_of_alberta(area_km2)
    if 0 < percent < 1:
        return "Area: less than 1% of Alberta"
    percent_rounded = int(round(percent))
    return f"Area: about {percent_rounded}% of Alberta"


def _build_facts(
    info: CountryInfo,
    fact_order: List[str],
    count: int,
    area_km2_override: Optional[float] = None,
) -> List[str]:
    area_km2 = area_km2_override if area_km2_override is not None else info.area_km2
    candidates: Dict[str, Optional[str]] = {
        "capital": f"Capital: {info.capital}" if info.capital else None,
        "continent": f"Continent: {info.continent}" if info.continent else None,
        "currency": f"Currency: {info.currencies}" if info.currencies else None,
        "languages": f"Languages: {info.languages}" if info.languages else None,
        "population": f"Population: {_format_population(info.population)}" if info.population else None,
        "area": _format_area_vs_alberta(area_km2) if area_km2 else None,
    }

    facts: List[str] = []
    for key in fact_order:
        text = candidates.get(key)
        if text and text not in facts:
            facts.append(text)
        if len(facts) >= count:
            break
    return facts


def _split_listish(value: Optional[str]) -> List[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    # Preserve original order but de-dupe
    seen: set[str] = set()
    out: List[str] = []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _build_display_lines(
    *,
    capital: Optional[str],
    population_display: Optional[str],
    area_line: Optional[str],
    continent: Optional[str],
    currency_list: List[str],
    max_lines: int = 3,
) -> List[str]:
    """
    Build a small set of lines intended to be shown in a compact info box.

    Primary set: capital, population, area (vs Alberta).
    Fallbacks: continent, currency.
    """
    candidates: List[Optional[str]] = [
        f"Capital: {capital}" if capital else None,
        f"Population: {population_display}" if population_display else None,
        area_line,
        f"Continent: {continent}" if continent else None,
        f"Currency: {currency_list[0]}" if currency_list else None,
    ]
    lines: List[str] = []
    for line in candidates:
        if not line:
            continue
        if line in lines:
            continue
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return lines


def _iter_alpha2_codes(flags: Dict[str, Any]) -> List[str]:
    codes: List[str] = []
    for code in flags.keys():
        if len(code) == 2 and code.isalpha():
            codes.append(code.lower())
    return sorted(set(codes))


def _build_sparql_query(codes_upper: Iterable[str]) -> str:
    values = " ".join(f"\"{c}\"" for c in codes_upper)
    return f"""
SELECT
  ?code
  ?country
  ?countryLabel
  ?capitalLabel
  ?population
  ?area
  ?continentLabel
  (GROUP_CONCAT(DISTINCT ?languageLabel; separator=", ") AS ?languages)
  (GROUP_CONCAT(DISTINCT ?currencyLabel; separator=", ") AS ?currencies)
WHERE {{
  VALUES ?code {{ {values} }}
  ?country wdt:P297 ?code .
  OPTIONAL {{ ?country wdt:P36 ?capital . }}
  OPTIONAL {{ ?country wdt:P1082 ?population . }}
  OPTIONAL {{ ?country wdt:P2046 ?area . }}
  OPTIONAL {{ ?country wdt:P30 ?continent . }}
  OPTIONAL {{ ?country wdt:P37 ?language . }}
  OPTIONAL {{ ?country wdt:P38 ?currency . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
GROUP BY ?code ?country ?countryLabel ?capitalLabel ?population ?area ?continentLabel
""".strip()


def fetch_wikidata_country_info(codes_alpha2_lower: List[str]) -> Dict[str, CountryInfo]:
    codes_upper = [c.upper() for c in codes_alpha2_lower]
    sparql = _build_sparql_query(codes_upper)
    url = f"{WIKIDATA_SPARQL_ENDPOINT}?{urllib.parse.urlencode({'format': 'json', 'query': sparql})}"

    LOGGER.info("Querying Wikidata for %d ISO alpha-2 codes", len(codes_upper))
    data = _http_get_json(url)
    bindings = data.get("results", {}).get("bindings", [])

    info_by_code: Dict[str, CountryInfo] = {}
    for b in bindings:
        code_upper = b["code"]["value"]
        code = code_upper.lower()

        country_uri = b["country"]["value"]
        qid = _qid_from_entity_uri(country_uri)

        def get_str(key: str) -> Optional[str]:
            v = b.get(key)
            if not v:
                return None
            s = v.get("value")
            return s if s else None

        def get_int(key: str) -> Optional[int]:
            s = get_str(key)
            if not s:
                return None
            try:
                return int(float(s))
            except ValueError:
                return None

        def get_float(key: str) -> Optional[float]:
            s = get_str(key)
            if not s:
                return None
            try:
                return float(s)
            except ValueError:
                return None

        info_by_code[code] = CountryInfo(
            code=code,
            qid=qid,
            label=get_str("countryLabel"),
            capital=get_str("capitalLabel"),
            continent=get_str("continentLabel"),
            currencies=get_str("currencies"),
            languages=get_str("languages"),
            population=get_int("population"),
            area_km2=get_float("area"),
        )

    return info_by_code


def fetch_wikidata_area_by_qid(qids: Iterable[str]) -> Dict[str, float]:
    qids_clean = [qid.strip() for qid in qids if qid and qid.strip()]
    qids_clean = [qid for qid in qids_clean if qid.startswith("Q") and qid[1:].isdigit()]
    if not qids_clean:
        return {}

    values = " ".join(f"wd:{qid}" for qid in sorted(set(qids_clean)))
    sparql = f"""
SELECT ?entity ?area WHERE {{
  VALUES ?entity {{ {values} }}
  ?entity wdt:P2046 ?area .
}}
""".strip()
    url = f"{WIKIDATA_SPARQL_ENDPOINT}?{urllib.parse.urlencode({'format': 'json', 'query': sparql})}"
    data = _http_get_json(url)
    bindings = data.get("results", {}).get("bindings", [])
    out: Dict[str, float] = {}
    for b in bindings:
        entity_uri = b["entity"]["value"]
        qid = _qid_from_entity_uri(entity_uri)
        try:
            out[qid] = float(b["area"]["value"])
        except ValueError:
            continue
    return out


def fetch_restcountries_area_by_code() -> Dict[str, float]:
    """
    Returns alpha-2 lowercase code -> area_km2.

    REST Countries returns `area` in km² for most entries and matches many common
    "country area" figures (e.g. France ~551,695 km²).
    """
    request = urllib.request.Request(
        RESTCOUNTRIES_ALL_FIELDS_URL,
        headers={
            "User-Agent": "InkyPi (flags restcountries enrich; +https://github.com/mcdarby/InkyPi)",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    data = json.loads(payload.decode("utf-8"))
    area_by_code: Dict[str, float] = {}
    if not isinstance(data, list):
        return area_by_code
    for item in data:
        if not isinstance(item, dict):
            continue
        code = item.get("cca2")
        area = item.get("area")
        if not isinstance(code, str) or len(code) != 2:
            continue
        if not isinstance(area, (int, float)):
            continue
        area_by_code[code.lower()] = float(area)
    return area_by_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich flags.json with Wikidata facts")
    parser.add_argument(
        "--in-file",
        default="flags.json",
        help="Input manifest path (default: flags.json)",
    )
    parser.add_argument(
        "--out-file",
        default="flags.json",
        help="Output manifest path (default: flags.json, overwrites in place)",
    )
    parser.add_argument(
        "--facts-count",
        type=int,
        default=3,
        help="Number of fact strings to attach per flag (default: 3)",
    )
    parser.add_argument(
        "--facts-order",
        nargs="+",
        default=FACT_KEYS_DEFAULT,
        help=f"Fact priority order (default: {' '.join(FACT_KEYS_DEFAULT)})",
    )
    parser.add_argument(
        "--area-provider",
        choices=["auto", "wikidata", "restcountries"],
        default="wikidata",
        help="Where to source area km² from for the Alberta comparison (default: wikidata)",
    )
    parser.add_argument(
        "--area-entity-overrides",
        default="",
        help="Optional JSON file mapping alpha-2 code -> Wikidata QID to source area from",
    )
    parser.add_argument(
        "--area-overrides",
        default="",
        help="Optional JSON file mapping alpha-2 code -> area_km2 to override area comparisons",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    in_path = Path(args.in_file).resolve()
    out_path = Path(args.out_file).resolve()
    if not in_path.exists():
        LOGGER.error("Input file not found: %s", in_path)
        return 2

    manifest: Dict[str, Any] = json.loads(in_path.read_text(encoding="utf-8"))
    flags: Dict[str, Any] = manifest.get("flags", {})
    if not isinstance(flags, dict) or not flags:
        LOGGER.error("No flags found in %s", in_path)
        return 2

    codes = _iter_alpha2_codes(flags)
    info_by_code = fetch_wikidata_country_info(codes)

    wants_area = any(k.strip() == "area" for k in args.facts_order)
    area_by_code: Dict[str, float] = {}
    if wants_area and args.area_provider == "restcountries":
        LOGGER.info("Fetching areas from REST Countries for Alberta comparisons")
        area_by_code = fetch_restcountries_area_by_code()

    area_entity_overrides: Dict[str, str] = dict(DEFAULT_AREA_ENTITY_OVERRIDES)
    if wants_area and args.area_entity_overrides:
        overrides_path = Path(args.area_entity_overrides).resolve()
        overrides_data = json.loads(overrides_path.read_text(encoding="utf-8"))
        if isinstance(overrides_data, dict):
            for k, v in overrides_data.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    continue
                code = k.strip().lower()
                qid = v.strip()
                if len(code) != 2 or not code.isalpha():
                    continue
                if not (qid.startswith("Q") and qid[1:].isdigit()):
                    continue
                area_entity_overrides[code] = qid

    qids_needed = [qid for qid in area_entity_overrides.values()]
    area_by_qid: Dict[str, float] = {}
    if wants_area and args.area_provider in {"auto", "wikidata"} and qids_needed:
        LOGGER.info("Fetching Wikidata areas for %d override entities", len(set(qids_needed)))
        area_by_qid = fetch_wikidata_area_by_qid(qids_needed)

    area_overrides_by_code: Dict[str, float] = {}
    if wants_area and args.area_overrides:
        overrides_path = Path(args.area_overrides).resolve()
        overrides_data = json.loads(overrides_path.read_text(encoding="utf-8"))
        if isinstance(overrides_data, dict):
            for k, v in overrides_data.items():
                if not isinstance(k, str):
                    continue
                code = k.strip().lower()
                if len(code) != 2 or not code.isalpha():
                    continue
                if isinstance(v, (int, float)):
                    area_overrides_by_code[code] = float(v)

    enriched = 0
    for code, entry in flags.items():
        if not (len(code) == 2 and code.isalpha()):
            continue
        info = info_by_code.get(code.lower())
        if not info:
            continue

        effective_area_km2: Optional[float] = None
        if wants_area:
            effective_area_km2 = (
                area_overrides_by_code.get(code.lower())
                or (
                    area_by_qid.get(area_entity_overrides.get(code.lower(), ""))
                    if args.area_provider in {"auto", "wikidata"}
                    else None
                )
                or area_by_code.get(code.lower())
                or info.area_km2
            )
            if effective_area_km2:
                entry["area_km2"] = int(round(effective_area_km2))
                entry["area_vs_alberta_pct"] = round(_area_percent_of_alberta(effective_area_km2), 2)
                entry["area_vs_alberta"] = _format_area_vs_alberta(effective_area_km2).replace("Area: ", "")
                entry["area_vs_alberta_line"] = _format_area_vs_alberta(effective_area_km2)

        if info.capital:
            entry["capital"] = info.capital
        if info.continent:
            entry["continent"] = info.continent
        if info.population:
            entry["population"] = info.population
            entry["population_display"] = _format_population(info.population)

        currency_list = _split_listish(info.currencies)
        language_list = _split_listish(info.languages)
        if currency_list:
            entry["currencies"] = currency_list
        if language_list:
            entry["languages"] = language_list

        entry["wikidata"] = {
            "qid": info.qid,
            "url": f"https://www.wikidata.org/wiki/{info.qid}",
        }

        # Normalize flag paths into a single object for wiring.
        entry["flag"] = {
            "png": entry.get("png"),
            "svg": entry.get("svg"),
            "code": code.lower(),
        }

        entry["facts"] = _build_facts(
            info=info,
            fact_order=[k.strip() for k in args.facts_order if k.strip()],
            count=max(0, int(args.facts_count)),
            area_km2_override=(
                area_overrides_by_code.get(code.lower())
                or (
                    area_by_qid.get(area_entity_overrides.get(code.lower(), ""))
                    if wants_area and args.area_provider in {"auto", "wikidata"}
                    else None
                )
                or area_by_code.get(code.lower())
                if wants_area
                else None
            ),
        )

        entry["display"] = {
            "title": entry.get("name"),
            "lines": _build_display_lines(
                capital=entry.get("capital"),
                population_display=entry.get("population_display"),
                area_line=entry.get("area_vs_alberta_line"),
                continent=entry.get("continent"),
                currency_list=currency_list,
                max_lines=3,
            ),
        }
        enriched += 1

    meta = manifest.setdefault("_meta", {})
    meta["enriched_at"] = datetime.now(timezone.utc).isoformat()
    meta["facts_source"] = "Wikidata (CC0)"
    meta["facts_provider"] = "query.wikidata.org"
    meta["facts_keys"] = [k.strip() for k in args.facts_order if k.strip()]
    meta["facts_count"] = int(args.facts_count)
    meta["area_reference"] = "Alberta"
    meta["area_reference_km2"] = ALBERTA_AREA_KM2
    meta["area_provider"] = args.area_provider
    if wants_area and args.area_provider == "restcountries" and area_by_code:
        meta["area_provider_url"] = RESTCOUNTRIES_ALL_FIELDS_URL
    if wants_area and area_overrides_by_code and args.area_overrides:
        meta["area_overrides_file"] = args.area_overrides
    if wants_area and args.area_provider in {"auto", "wikidata"} and area_entity_overrides:
        meta["area_entity_overrides"] = area_entity_overrides
    if wants_area and args.area_entity_overrides:
        meta["area_entity_overrides_file"] = args.area_entity_overrides
    meta["display_layout_hint"] = {
        "canvas_px": [560, 480],
        "lines_max": 3,
        "recommended_fields": ["capital", "population", "area_vs_alberta"],
    }

    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s (enriched %d entries)", out_path, enriched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
