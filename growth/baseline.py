from typing import Dict, List, Tuple

def evaluate_baseline_rules(current_z: Dict[str, float]) -> Tuple[str, List[str]]:
    """
    Evaluates standard WHO rules (R1-R4), margin watch rules (R5-R6),
    and returns (Status, Fired_Rule_IDs).
    """
    fired = []
    
    # Conventional Critical Thresholds (Potential Growth Anomaly)
    if current_z.get("waz", 0) < -2.0:
        fired.append("R1")
    if current_z.get("haz", 0) < -2.0:
        fired.append("R2")
    if abs(current_z.get("bmiz", 0)) > 2.0:
        fired.append("R3")
    if abs(current_z.get("hcz", 0)) > 2.0:
        fired.append("R4")
        
    if fired:
        return "Potential Growth Anomaly", fired
        
    # Project-defined Watch Margins
    if 1.0 < current_z.get("bmiz", 0) <= 2.0:
        fired.append("R5")
        
    near_cutoff = any(-2.0 <= current_z.get(k, 0) < -1.5 for k in ("waz", "haz", "bmiz", "hcz")) or \
                  (1.5 < current_z.get("hcz", 0) <= 2.0)
    if near_cutoff:
        fired.append("R6")
        
    if fired:
        return "Watch", fired
        
    return "Normal", [] 
