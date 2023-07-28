from enum import Enum

INFECTION_STATUS = "infection_status"
POWER_USAGE = "power_usage"
HOUSEHOLD_INDEX = "h_index"
ACTIVATED = "activated"
HOUSEHOLD_APPLIANCE = "appliances"
WILL_ACT = "will_act"
P_S = "p_s"
P_I = "p_i"
P_R = "p_r"
class InfectionStatus(Enum):
    SUSCEPTIBLE = 0
    FACT_CHECKER = 1
    BELIEVER = 2
