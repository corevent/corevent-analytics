"""Enums mirrored from api-corevent entities."""

from __future__ import annotations

from enum import StrEnum


class EventCategory(StrEnum):
    MUSIC = "music"
    SPORTS = "sports"
    TECH = "tech"
    BUSINESS = "business"
    EDUCATION = "education"
    ART_CULTURE = "art_culture"
    GASTRONOMY = "gastronomy"
    HEALTH_WELLNESS = "health_wellness"
    FAMILY_KIDS = "family_kids"
    RELIGIOUS_SPIRITUAL = "religious_spiritual"
    GAMES = "games"
    COMMUNITY_SOCIAL = "community_social"
    FASHION_BEAUTY = "fashion_beauty"
    OTHER = "other"


class EventLocationType(StrEnum):
    ONLINE = "online"
    IN_PERSON = "in_person"
    HYBRID = "hybrid"


class EventStatus(StrEnum):
    DRAFT = "draft"
    OPENED = "opened"
    GOING = "going"
    CANCELED = "canceled"
    FINISHED = "finished"


class OrderStatus(StrEnum):
    PENDING = "pending"
    PAID = "paid"
    CANCELLED = "cancelled"
    FAILED = "failed"
    PENDING_REFUND = "pending_refund"


class TicketStatus(StrEnum):
    PENDING = "pending"
    CHECKED_IN = "checked_in"
    CANCELLED = "cancelled"


EVENT_CATEGORIES: tuple[str, ...] = tuple(c.value for c in EventCategory)
EVENT_LOCATION_TYPES: tuple[str, ...] = tuple(t.value for t in EventLocationType)
EVENT_STATUSES: tuple[str, ...] = tuple(s.value for s in EventStatus)
ORDER_STATUSES: tuple[str, ...] = tuple(s.value for s in OrderStatus)
TICKET_STATUSES: tuple[str, ...] = tuple(s.value for s in TicketStatus)

# Co-occurrence affinities so Apriori finds meaningful rules on synthetic data.
CATEGORY_AFFINITIES: dict[str, tuple[str, ...]] = {
    EventCategory.MUSIC.value: (
        EventCategory.GASTRONOMY.value,
        EventCategory.ART_CULTURE.value,
        EventCategory.FASHION_BEAUTY.value,
    ),
    EventCategory.TECH.value: (
        EventCategory.BUSINESS.value,
        EventCategory.EDUCATION.value,
        EventCategory.GAMES.value,
    ),
    EventCategory.SPORTS.value: (
        EventCategory.HEALTH_WELLNESS.value,
        EventCategory.FAMILY_KIDS.value,
        EventCategory.COMMUNITY_SOCIAL.value,
    ),
    EventCategory.GASTRONOMY.value: (
        EventCategory.MUSIC.value,
        EventCategory.ART_CULTURE.value,
        EventCategory.COMMUNITY_SOCIAL.value,
    ),
    EventCategory.EDUCATION.value: (
        EventCategory.TECH.value,
        EventCategory.BUSINESS.value,
        EventCategory.RELIGIOUS_SPIRITUAL.value,
    ),
}
