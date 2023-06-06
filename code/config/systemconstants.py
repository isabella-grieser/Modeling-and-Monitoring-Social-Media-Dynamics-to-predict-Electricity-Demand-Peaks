from enum import Enum

INFECTION_STATUS = "infection_status"
POWER_USAGE = "power_usage"
HOUSEHOLD_INDEX = "h_index"
ACTIVATED = "activated"
HOUSEHOLD_APPLIANCE = "appliances"


class InfectionStatus(Enum):
    SUSCEPTIBLE = 0
    FACT_CHECKER = 1
    BELIEVER = 2
