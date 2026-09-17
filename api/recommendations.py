"""
Transparent Rule-Based Recommendation Engine for OTT Audience Segmentation.
Maps audience segments and user genre affinities to personalized content titles.
"""

from typing import List, Dict, Set


CATALOG_BY_GENRE: Dict[str, List[str]] = {
    "Action": [
        "Extraction Velocity",
        "Cyber Strike",
        "Shadow Protocol",
        "Apex Hunter",
        "Velocity Chase",
    ],
    "Sci-Fi": [
        "Interstellar Odyssey",
        "Quantum Nexus",
        "Nebula Drift",
        "Chronos Paradox",
        "Solaris Awakening",
    ],
    "Thriller": [
        "Midnight Intrigue",
        "Silent Witness",
        "The Blind Spot",
        "Frenzy Point",
        "Deception Point",
    ],
    "Comedy": [
        "Weekend Hangout",
        "Laugh Riot",
        "Campus Chaos",
        "Awkward Moments",
        "Office Mayhem",
    ],
    "Drama": [
        "Echoes of the Past",
        "Autumn Chronicles",
        "The Long Road",
        "City Lights",
        "Legacy of Truth",
    ],
    "Family": [
        "Adventure Island",
        "The Magic Whistle",
        "Paws & Whiskers",
        "Star Gazers",
        "Forest Friends",
    ],
    "Animation": [
        "Pixel Quest",
        "Dragon Whisperer",
        "Toy Kingdom",
        "Cloud Riders",
        "Robot Odyssey",
    ],
    "Romance": [
        "Whispers in the Rain",
        "Summer Sunset",
        "Letters to You",
        "Parisian Nights",
        "Second Chances",
    ],
    "Documentary": [
        "Planet Earth Mysteries",
        "The AI Frontier",
        "Deep Ocean Wonders",
        "Ancient Echoes",
        "Wild Expeditions",
    ],
}

DEFAULT_POPULAR = [
    "Echoes of the Past",
    "Weekend Hangout",
    "Extraction Velocity",
    "Adventure Island",
]


def generate_recommendations(
    segment_name: str,
    recommended_genres: List[str],
    user_top_genres: List[str],
    limit: int = 4,
) -> List[str]:
    """
    Generates tailored recommendations based on predicted segment strategy
    and candidate's specific top genres.
    """
    results: List[str] = []
    seen: Set[str] = set()

    # Priority 1: User's top genres that intersect with segment recommended genres
    overlap_genres = [g for g in user_top_genres if g in recommended_genres]
    for g in overlap_genres:
        for title in CATALOG_BY_GENRE.get(g, []):
            if title not in seen:
                results.append(title)
                seen.add(title)
                if len(results) >= limit:
                    return results

    # Priority 2: Primary recommended genres from segment profile
    for g in recommended_genres:
        for title in CATALOG_BY_GENRE.get(g, []):
            if title not in seen:
                results.append(title)
                seen.add(title)
                if len(results) >= limit:
                    return results

    # Priority 3: User's top genres even if not in segment's top list
    for g in user_top_genres:
        for title in CATALOG_BY_GENRE.get(g, []):
            if title not in seen:
                results.append(title)
                seen.add(title)
                if len(results) >= limit:
                    return results

    # Priority 4: Fallback popular catalog titles
    for title in DEFAULT_POPULAR:
        if title not in seen:
            results.append(title)
            seen.add(title)
            if len(results) >= limit:
                return results

    return results
