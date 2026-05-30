"""
PHONE SERVICE (ADVANCED)
========================
Tracks phone numbers using `phonenumbers` plus timezone, geocoding,
and carrier data. Returns rich, accurate metadata even for partially
formatted inputs.

Install:  pip install phonenumbers
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger("J.A.R.V.I.S")

# Reasonable display names for common ISO regions.
_COUNTRY_MAP = {
    "US": "United States", "GB": "United Kingdom", "IN": "India",
    "AU": "Australia", "CA": "Canada", "DE": "Germany", "FR": "France",
    "JP": "Japan", "CN": "China", "BR": "Brazil", "MX": "Mexico",
    "RU": "Russia", "KR": "South Korea", "NG": "Nigeria", "ZA": "South Africa",
    "PK": "Pakistan", "BD": "Bangladesh", "ID": "Indonesia", "TR": "Turkey",
    "SA": "Saudi Arabia", "AE": "United Arab Emirates", "EG": "Egypt",
    "IT": "Italy", "ES": "Spain", "NL": "Netherlands", "PH": "Philippines",
    "TH": "Thailand", "MY": "Malaysia", "SG": "Singapore", "HK": "Hong Kong",
    "PL": "Poland", "SE": "Sweden", "NO": "Norway", "DK": "Denmark",
    "FI": "Finland", "CH": "Switzerland", "AT": "Austria", "BE": "Belgium",
    "NZ": "New Zealand", "AR": "Argentina", "CL": "Chile", "CO": "Colombia",
    "VN": "Vietnam", "UA": "Ukraine", "IL": "Israel", "IR": "Iran",
    "IQ": "Iraq", "ET": "Ethiopia", "KE": "Kenya", "GH": "Ghana",
    "NP": "Nepal", "LK": "Sri Lanka", "MM": "Myanmar", "AF": "Afghanistan",
    "IE": "Ireland", "PT": "Portugal", "GR": "Greece", "CZ": "Czechia",
    "RO": "Romania", "HU": "Hungary",
}


def _candidate_regions(raw: str) -> List[Optional[str]]:
    """Pick parse regions based on the leading digits."""
    digits = re.sub(r"\D", "", raw or "")
    # Always try "no region" first so a leading + is honoured.
    out: List[Optional[str]] = [None]
    if not raw.startswith("+"):
        if digits.startswith("1"):
            out += ["US", "CA"]
        elif digits.startswith("91") or len(digits) == 10:
            out += ["IN"]
        elif digits.startswith("44"):
            out += ["GB"]
        elif digits.startswith("61"):
            out += ["AU"]
        else:
            out += ["US", "IN", "GB"]
    return out


def track_phone(number_raw: str) -> dict:
    try:
        import phonenumbers
        from phonenumbers import (
            geocoder,
            carrier as pn_carrier,
            timezone as pn_timezone,
            number_type as pn_number_type,
            PhoneNumberType,
        )
    except ImportError:
        return {
            "valid": False,
            "error": "phonenumbers not installed. Run: pip install phonenumbers",
            "raw": number_raw,
        }

    raw = (number_raw or "").strip()
    if not raw:
        return {"valid": False, "error": "No number provided.", "raw": raw}

    # Try every plausible region until we get a valid parse.
    parsed = None
    parse_err: Optional[str] = None
    for region in _candidate_regions(raw):
        try:
            p = phonenumbers.parse(raw, region)
            if phonenumbers.is_possible_number(p):
                parsed = p
                if phonenumbers.is_valid_number(p):
                    break
        except phonenumbers.NumberParseException as exc:
            parse_err = str(exc)
            continue

    if parsed is None:
        return {
            "valid": False,
            "error": f"Could not parse '{raw}' as a phone number." + (
                f" ({parse_err})" if parse_err else ""
            ),
            "raw": raw,
        }

    is_valid = phonenumbers.is_valid_number(parsed)
    is_possible = phonenumbers.is_possible_number(parsed)

    fmt_e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    fmt_intl = phonenumbers.format_number(
        parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL
    )
    fmt_natl = phonenumbers.format_number(
        parsed, phonenumbers.PhoneNumberFormat.NATIONAL
    )

    region_code = phonenumbers.region_code_for_number(parsed) or ""
    country = _COUNTRY_MAP.get(region_code.upper(), region_code or "Unknown")

    # The geocoder returns the most specific area description it has
    # (city / state / region). Fall back to country if empty.
    location = geocoder.description_for_number(parsed, "en") or country
    carrier_name = pn_carrier.name_for_number(parsed, "en") or "Unknown"
    timezones = list(pn_timezone.time_zones_for_number(parsed) or [])

    ntype = pn_number_type(parsed)
    type_labels = {
        PhoneNumberType.MOBILE: "Mobile",
        PhoneNumberType.FIXED_LINE: "Fixed Line",
        PhoneNumberType.FIXED_LINE_OR_MOBILE: "Fixed Line or Mobile",
        PhoneNumberType.TOLL_FREE: "Toll Free",
        PhoneNumberType.PREMIUM_RATE: "Premium Rate",
        PhoneNumberType.SHARED_COST: "Shared Cost",
        PhoneNumberType.VOIP: "VoIP",
        PhoneNumberType.PERSONAL_NUMBER: "Personal Number",
        PhoneNumberType.PAGER: "Pager",
        PhoneNumberType.UAN: "UAN",
        PhoneNumberType.VOICEMAIL: "Voicemail",
        PhoneNumberType.UNKNOWN: "Unknown",
    }

    # Build a Google Maps link for the best location string we have.
    map_query = location or country
    map_url = (
        "https://www.google.com/maps/search/?api=1&query="
        + re.sub(r"\s+", "+", map_query)
        if map_query
        else ""
    )

    return {
        "valid": is_valid,
        "possible": is_possible,
        "raw": raw,
        "formatted_e164": fmt_e164,
        "formatted_international": fmt_intl,
        "formatted_national": fmt_natl,
        "country_code": "+" + str(parsed.country_code),
        "country_name": country,
        "region_code": region_code,
        "location": location,
        "carrier": carrier_name,
        "timezones": timezones,
        "number_type": type_labels.get(ntype, "Unknown"),
        "map_url": map_url,
    }


def format_phone_result_for_chat(result: dict) -> str:
    if not result.get("valid") and not result.get("possible"):
        return result.get("error") or (
            "The number " + result.get("raw", "") + " does not appear to be valid."
        )

    lines = [
        "Phone Number: " + result["formatted_international"],
        "Country: " + result["country_name"] + " (" + result["country_code"] + ")",
    ]
    if result.get("location") and result["location"] != result["country_name"]:
        lines.append("Location: " + result["location"])
    if result.get("carrier") and result["carrier"] != "Unknown":
        lines.append("Carrier: " + result["carrier"])
    lines.append("Type: " + result.get("number_type", "Unknown"))
    if result.get("timezones"):
        lines.append("Timezone: " + ", ".join(result["timezones"]))
    lines.append("E.164: " + result["formatted_e164"])
    lines.append("National: " + result["formatted_national"])
    if result.get("map_url"):
        lines.append("Map: " + result["map_url"])
    if not result.get("valid"):
        lines.append(
            "Note: number is possible but not confirmed valid for this region."
        )
    return "\n".join(lines)
