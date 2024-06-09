from enum import Enum

# Marker for pseudo class indicating that there is no relationship between
# the entity pair under investigation
NO_RELATION_CLASS = "_no-relation_"

# Mapping from dataset specific entity type names to harmonized ones
ENTITY_TYPE_TO_DEFAULT_NAME = {
    "chemical": "drug",
    "gene": "gene",
    "gene-n": "gene",
    "gene-y": "gene",
    "disease": "disease",
}


# Strategies to mark the entities under investigation
class EntityMarker(Enum):
    SPECIAL_TOKEN = ("[HEAD-S]", "[HEAD-E]", "[TAIL-S]", "[TAIL-E]")
    NORMAL_MARKER = ("@", "@", "$", "$")

