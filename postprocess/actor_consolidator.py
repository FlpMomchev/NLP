from typing import Dict, List, Set, Tuple
from difflib import SequenceMatcher
import re

# Banking domain mappings as module-level constants
KNOWN_MAPPINGS = {
    # Banking specific
    "KYC": "KYC Team",
    "AML": "Compliance Team",
    "Compliance": "Compliance Team",
    "Risk": "Risk Team",
    "Fraud": "Risk Team",
    "Core Banking": "Core Banking",
    "Digital Channel": "Digital Channel",
    "Customer Service": "Customer Service",
    "External Vendor": "External Vendor",

    # Generic business
    "PM": "Project Manager",
    "PSC": "Product Steering Committee",
    "R&D": "R&D Team",
    "QA": "Quality Assurance",
    "Process Owner": "Process Owner"
}

# Consolidation rules for banking domain
CONSOLIDATION_RULES = {
    "KYC Team": [
        "kyc", "know your customer", "customer verification",
        "identity verification", "due diligence"
    ],
    "Risk Team": [
        "risk", "fraud", "security", "threat", "anomaly detection",
        "risk engine", "fraud detection", "security team"
    ],
    "Compliance Team": [
        "compliance", "regulatory", "audit", "governance",
        "aml", "anti money laundering", "regulatory affairs"
    ],
    "Core Banking": [
        "core banking", "banking system", "core system", "banking core",
        "account management", "transaction processing", "banking engine"
    ],
    "Digital Channel": [
        "digital", "online", "web", "portal", "channel", "digital platform",
        "online banking", "digital service", "web portal"
    ],
    "Customer Service": [
        "customer", "client", "user", "customer support", "customer success",
        "customer service", "client service", "user support"
    ]
}

IGNORE_PREFIXES = {"the", "a", "an", "our", "my", "your", "their"}

DOMAIN_ABBREVIATIONS = {
    "kyc": "know your customer",
    "aml": "anti money laundering",
    "cdd": "customer due diligence",
    "edd": "enhanced due diligence"
}


def remove_articles(text: str) -> str:
    """
    Remove common articles and prefixes from actor names.

    Cleans actor names by removing leading articles that don't contribute
    to semantic meaning during matching and consolidation.

    Args:
        text (str): Actor name to clean

    Returns:
        str: Cleaned actor name without leading articles

    """
    words = text.split()
    if words and words[0].lower() in IGNORE_PREFIXES:
        return " ".join(words[1:])
    return text


def find_canonical_form(actor: str) -> str:
    """
    Determine the canonical form with banking domain priority.

    Applies domain-specific knowledge to map actor variations to their
    canonical forms. Uses a priority system: known mappings, consolidation
    rules, then business role detection.

    Args:
        actor (str): Original actor name to canonicalize

    Returns:
        str: Canonical form of the actor name

    """
    cleaned = remove_articles(actor)

    # Priority 1: Check known mappings
    for key, canonical in KNOWN_MAPPINGS.items():
        if key.lower() == cleaned.lower() or key in cleaned:
            return canonical

    # Priority 2: Check if it matches domain consolidation rules
    cleaned_lower = cleaned.lower()
    for canonical_name, keywords in CONSOLIDATION_RULES.items():
        if any(keyword in cleaned_lower for keyword in keywords):
            return canonical_name

    # Priority 3: Smart business role detection
    if "team" in cleaned.lower():
        return cleaned
    elif "manager" in cleaned.lower():
        return "Process Owner"
    elif "owner" in cleaned.lower():
        return "Process Owner"

    return cleaned


def are_abbreviation_related(actor1: str, actor2: str) -> bool:
    """
    Enhanced abbreviation detection between two actor names.

    Checks if two actor names represent the same entity using different
    abbreviation forms (e.g., "KYC" and "Know Your Customer").

    Args:
        actor1 (str): First actor name
        actor2 (str): Second actor name

    Returns:
        bool: True if actors are abbreviation-related, False otherwise

    """
    actor1_clean = remove_articles(actor1).lower()
    actor2_clean = remove_articles(actor2).lower()

    # Check known mappings
    for abbr, full in KNOWN_MAPPINGS.items():
        abbr_lower = abbr.lower()
        full_lower = full.lower()

        if (abbr_lower in actor1_clean and full_lower in actor2_clean) or \
                (abbr_lower in actor2_clean and full_lower in actor1_clean):
            return True

    # Check domain-specific abbreviations
    for abbr, full in DOMAIN_ABBREVIATIONS.items():
        if (abbr in actor1_clean and full in actor2_clean) or \
                (abbr in actor2_clean and full in actor1_clean):
            return True

    return False


def find_similar_actors(actor: str, all_actors: List[str], processed: Set[str]) -> List[str]:
    """
    Find similar actors with enhanced business logic.

    Identifies actor names that likely represent the same entity using
    multiple similarity detection methods: exact matching, containment,
    fuzzy string matching, and abbreviation detection.

    Args:
        actor (str): Target actor to find similarities for
        all_actors (List[str]): Complete list of actors to search
        processed (Set[str]): Set of already processed actors to skip

    Returns:
        List[str]: List of similar actor names including the original

    """
    similar = [actor]
    actor_clean = remove_articles(actor).lower()

    for other in all_actors:
        if other in processed or other == actor:
            continue

        other_clean = remove_articles(other).lower()

        # Exact match after cleaning
        if actor_clean == other_clean:
            similar.append(other)
            continue

        # One contains the other (with length check to avoid false positives)
        if len(actor_clean) > 3 and len(other_clean) > 3:
            if actor_clean in other_clean or other_clean in actor_clean:
                similar.append(other)
                continue

        similarity = SequenceMatcher(None, actor_clean, other_clean).ratio()
        if similarity > 0.90:
            similar.append(other)
            continue

        # Check for known abbreviation relationships
        if are_abbreviation_related(actor, other):
            similar.append(other)

    return similar


def consolidate_actors(actors: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Enhanced consolidation using banking domain knowledge.

    Performs two-step consolidation: domain-specific rule matching followed
    by fuzzy similarity matching. Creates bidirectional mappings between
    canonical forms and their variations.

    Args:
        actors (List[str]): List of actor names to consolidate

    Returns:
        Tuple[Dict[str, List[str]], Dict[str, str]]:
            - canonical_to_variations: Maps canonical names to their variations
            - variation_to_canonical: Maps each variation to its canonical form

    """
    canonical_to_variations = {}
    variation_to_canonical = {}
    processed = set()

    # Step 1: Apply domain-specific consolidation rules
    for canonical_name, keywords in CONSOLIDATION_RULES.items():
        matching_actors = []

        for actor in actors:
            if actor in processed:
                continue

            actor_lower = actor.lower().strip()

            # Check if actor matches any keyword for this canonical form
            if any(keyword in actor_lower for keyword in keywords):
                matching_actors.append(actor)
                processed.add(actor)

        if matching_actors:
            canonical_to_variations[canonical_name] = matching_actors
            for actor in matching_actors:
                variation_to_canonical[actor] = canonical_name

    # Step 2: Apply fuzzy matching for remaining actors
    remaining_actors = [a for a in actors if a not in processed]

    for actor in remaining_actors:
        if actor in processed:
            continue

        # Find canonical form
        canonical = find_canonical_form(actor)

        # Find similar actors
        similar_actors = find_similar_actors(actor, remaining_actors, processed)

        if canonical not in canonical_to_variations:
            canonical_to_variations[canonical] = []

        # Add all variations
        for similar in similar_actors:
            canonical_to_variations[canonical].append(similar)
            variation_to_canonical[similar] = canonical
            processed.add(similar)

    return canonical_to_variations, variation_to_canonical


def get_consolidation_stats(original_actors: List[str], canonical_mapping: Dict[str, List[str]]) -> Dict[str, any]:
    """
    Enhanced statistics calculation for consolidation results.

    Computes comprehensive metrics about the consolidation process including
    reduction percentages, average variations per canonical form, and identifies
    the most consolidated actor.

    Args:
        original_actors (List[str]): Original list of actors before consolidation
        canonical_mapping (Dict[str, List[str]]): Mapping of canonical forms to variations

    Returns:
        Dict[str, any]: Statistics dictionary containing:
            - original_count: Number of unique original actors
            - consolidated_count: Number of canonical forms
            - reduction_percentage: Percentage reduction achieved
            - average_variations_per_canonical: Average variations per canonical form
            - most_consolidated: Canonical form with most variations

    """
    original_unique = len(set(original_actors))
    consolidated_count = len(canonical_mapping)

    # Calculate reduction more accurately
    reduction = 0 if original_unique == 0 else round(
        (1 - consolidated_count / original_unique) * 100, 1
    )

    return {
        "original_count": original_unique,
        "consolidated_count": consolidated_count,
        "reduction_percentage": reduction,
        "average_variations_per_canonical": round(
            sum(len(v) for v in canonical_mapping.values()) / consolidated_count, 1
        ) if consolidated_count > 0 else 0,
        "most_consolidated": max(
            canonical_mapping.items(),
            key=lambda x: len(x[1])
        )[0] if canonical_mapping else None
    }


def apply_actor_consolidation(process_blocks: List[Dict]) -> Tuple[List[Dict], Dict[str, any]]:
    """
    Enhanced actor consolidation with better error handling.

    Main entry point for actor consolidation. Processes workflow blocks to
    extract, consolidate, and update actor information. Handles multiple
    data structures and provides comprehensive consolidation statistics.

    This function:
    1. Extracts all actor names from process blocks and steps
    2. Applies domain-specific consolidation rules
    3. Updates all references with canonical forms
    4. Provides detailed consolidation statistics

    Args:
        process_blocks (List[Dict]): List of process blocks containing actor information

    Returns:
        Tuple[List[Dict], Dict[str, any]]:
            - Updated process blocks with consolidated actors
            - Consolidation metadata including mappings and statistics

    """

    # Collect all unique actors more robustly
    all_actors = []

    for block in process_blocks:
        # Handle different block structures
        if "primary_actors" in block:
            all_actors.extend(block["primary_actors"])

        # Handle steps structure
        for step in block.get("steps", []):
            if "primary_actors" in step:
                all_actors.extend(step["primary_actors"])
            elif "actor" in step:  # Handle single actor format
                all_actors.append(step["actor"])

    # Remove None values and empty strings
    all_actors = [actor for actor in all_actors if actor and actor.strip()]

    # Remove duplicates while preserving order
    unique_actors = list(dict.fromkeys(all_actors))

    if not unique_actors:
        # No actors found - return empty consolidation
        return process_blocks, {
            "actor_resolution": {
                "mappings": {},
                "variation_to_canonical": {},
                "statistics": {
                    "original_count": 0,
                    "consolidated_count": 0,
                    "reduction_percentage": 0,
                    "average_variations_per_canonical": 0
                }
            }
        }

    # Consolidate actors
    canonical_mapping, variation_mapping = consolidate_actors(unique_actors)

    # Update all blocks with canonical actors
    for block in process_blocks:
        # Update block actors
        if "primary_actors" in block:
            block["primary_actors"] = [
                variation_mapping.get(actor, actor)
                for actor in block["primary_actors"]
                if actor  # Filter out None/empty
            ]
            # Remove duplicates
            block["primary_actors"] = list(dict.fromkeys(block["primary_actors"]))

        # Update step actors
        for step in block.get("steps", []):
            if "primary_actors" in step:
                step["primary_actors"] = [
                    variation_mapping.get(actor, actor)
                    for actor in step["primary_actors"]
                    if actor
                ]
                step["primary_actors"] = list(dict.fromkeys(step["primary_actors"]))
            elif "actor" in step and step["actor"]:
                step["actor"] = variation_mapping.get(step["actor"], step["actor"])

    # Get statistics
    stats = get_consolidation_stats(unique_actors, canonical_mapping)

    return process_blocks, {
        "actor_resolution": {
            "mappings": canonical_mapping,
            "variation_to_canonical": variation_mapping,
            "statistics": stats
        }
    }