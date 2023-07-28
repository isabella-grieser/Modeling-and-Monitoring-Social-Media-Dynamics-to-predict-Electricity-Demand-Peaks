from enum import Enum

INFECTION_STATUS = "infection_status"
POWER_USAGE = "power_usage"
HOUSEHOLD_INDEX = "h_index"
ACTIVATED = "activated"
HOUSEHOLD_APPLIANCE = "appliances"
WILL_ACT = "will_act"
PREV_STATE = "prev_state"
P_S = "p_s"
P_I = "p_i"
P_R = "p_r"


class InfectionStatus(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
