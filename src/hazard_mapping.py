HAZARD_LEVELS = {
    13: "CAUTION: Yield",
    14: "HIGH RISK: Stop",
    17: "DANGER: No Entry",
    18: "CAUTION: General Caution",
    20: "CAUTION: Dangerous Curve Right",
    21: "CAUTION: Double Curve",
    22: "CAUTION: Bumpy Road",
    23: "CAUTION: Slippery Road",
    25: "CAUTION: Road Work",
    26: "CAUTION: Traffic Signals",
    27: "CAUTION: Pedestrians",
    28: "CAUTION: Children Crossing",
    29: "CAUTION: Bicycles Crossing",
    30: "CAUTION: Ice or Snow",
    31: "CAUTION: Wild Animals Crossing",
}

def get_hazard_warning(class_id):
    return HAZARD_LEVELS.get(class_id, "Normal Sign")

