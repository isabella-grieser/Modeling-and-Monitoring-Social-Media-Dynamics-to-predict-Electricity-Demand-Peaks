from enum import Enum

INFECTION_ATTR = "infection_status"

POWER_USAGE_ATTR = "power_usage"


class InfectionStatus(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2
