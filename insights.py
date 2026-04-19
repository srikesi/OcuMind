"""
insights.py — Holistic Clinical Insight Engine.
Extracts clinical features and generates intuitive, risk-scaled reports.
"""
import numpy as np
from typing import List, Dict

class InsightEngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def analyze_session(self, frames: List[Dict], mode: str) -> Dict:
        if not frames:
            return {"error": "No data to analyze"}

        features = self._extract_features(frames)
        report = self._generate_report(features)

        return {
            "features": features,
            "report": report
        }

    def _extract_features(self, frames: List[Dict]) -> Dict:
        gaze_x = np.array([f['gaze_x'] for f in frames])
        gaze_y = np.array([f['gaze_y'] for f in frames])
        target_x = np.array([f['target_x'] for f in frames])
        target_y = np.array([f['target_y'] for f in frames])
        errors = np.sqrt((gaze_x - target_x)**2 + (gaze_y - target_y)**2)
        
        accel = np.diff(np.sqrt(np.diff(gaze_x)**2 + np.diff(gaze_y)**2))
        jerkiness = np.std(accel) if len(accel) > 0 else 0

        split = len(errors) // 5
        fatigue_index = np.mean(errors[-split:]) - np.mean(errors[:split]) if split > 0 else 0

        h_err = np.mean(np.abs(gaze_x - target_x))
        v_err = np.mean(np.abs(gaze_y - target_y))

        return {
            "jerkiness": float(jerkiness),
            "fatigue_index": float(fatigue_index),
            "h_err": float(h_err),
            "v_err": float(v_err),
            "mean_error": float(np.mean(errors))
        }

    def _generate_report(self, feat: Dict) -> Dict:
        # Calculate Risk Percentages using Exponential Scaling 
        # (Keeps numbers very small unless variables are extreme)
        tremor_risk = min(99.9, max(0.1, (feat['jerkiness'] / 0.1) ** 4 * 100))
        fatigue_risk = min(99.9, max(0.1, (max(0, feat['fatigue_index']) / 0.3) ** 4 * 100))
        
        ratio = max(feat['h_err'] / (feat['v_err'] + 1e-6), feat['v_err'] / (feat['h_err'] + 1e-6))
        asym_risk = min(99.9, max(0.1, ((ratio - 1.0) / 2.0) ** 4 * 100))

        highest_risk = max(tremor_risk, fatigue_risk, asym_risk)

        # 1. Primary Conclusion (The BIG thing at the top)
        if highest_risk > 70:
            conclusion = {
                "status": "risk",
                "title": "Clinical Warning Detected",
                "subtitle": "Your tracking metrics show extreme deviations from normal eye-movement baselines. Please review the specific risk analysis below."
            }
        elif highest_risk > 15:
            conclusion = {
                "status": "warning",
                "title": "Moderate Tracking Deviations",
                "subtitle": "Your eyes showed some difficulty keeping up, which is very common with screen fatigue, lack of sleep, or general eye strain."
            }
        else:
            conclusion = {
                "status": "good",
                "title": "You have great tracking and vision!",
                "subtitle": "Your eyes followed the target smoothly, consistently, and evenly without getting tired."
            }

        # 2. Intuitive Stats Breakdown (The middle)
        stats = []
        stats.append({
            "name": "Smoothness", 
            "desc": "Your eyes glided smoothly to follow the dot without unnecessary jumping." if tremor_risk < 15 else "Your eyes made quick stops and starts (jumps) rather than gliding smoothly.", 
            "color": "green" if tremor_risk < 15 else ("orange" if tremor_risk < 70 else "red")
        })
        stats.append({
            "name": "Endurance", 
            "desc": "You maintained a steady, accurate focus from start to finish without losing pace." if fatigue_risk < 15 else "You started off strong, but your focus began lagging near the end.", 
            "color": "green" if fatigue_risk < 15 else ("orange" if fatigue_risk < 70 else "red")
        })
        stats.append({
            "name": "Directional Balance", 
            "desc": "Your eyes tracked equally well in all directions." if asym_risk < 15 else "You had an easier time tracking the dot in one direction compared to the other.", 
            "color": "green" if asym_risk < 15 else ("orange" if asym_risk < 70 else "red")
        })

        # 3. Disease Risk Profiling (Smaller, only alarming if extreme)
        diseases = []
        
        # Parkinson's / Tremor
        t_level = "high" if tremor_risk > 70 else ("med" if tremor_risk > 15 else "low")
        t_desc = "Extreme micro-saccadic jumping detected. This severe pattern is correlated with tremor disorders. Consult a specialist." if t_level == "high" else "Normal slight jitteriness, likely from screen strain or caffeine."
        diseases.append({"name": "Tremor / Parkinson's Marker", "pct": round(tremor_risk, 1), "level": t_level, "desc": t_desc})

        # MS / TBI
        f_level = "high" if fatigue_risk > 70 else ("med" if fatigue_risk > 15 else "low")
        f_desc = "Severe neurological tracking fatigue detected. Your visual system loses tracking ability rapidly." if f_level == "high" else "Standard visual endurance. Any minor lag is typical of normal eye fatigue."
        diseases.append({"name": "MS / TBI Fatigue Marker", "pct": round(fatigue_risk, 1), "level": f_level, "desc": f_desc})

        # Astigmatism
        a_level = "high" if asym_risk > 70 else ("med" if asym_risk > 15 else "low")
        a_desc = "Extreme directional asymmetry. This heavily implies uncorrected astigmatism or a muscular palsy." if a_level == "high" else "Eyes are well-balanced horizontally and vertically."
        diseases.append({"name": "Astigmatism Marker", "pct": round(asym_risk, 1), "level": a_level, "desc": a_desc})

        # 4. Recommendations
        recs = []
        if highest_risk < 15:
            recs.append("Keep up the great work! To challenge yourself, try increasing the speed to 'Fast' or using the 'Random jumps' mode.")
        if tremor_risk >= 15:
            recs.append("Reduce the target speed by 15-20% (use 'Slow' mode) in your next session to practice continuous, smooth movements.")
        if fatigue_risk >= 15:
            recs.append("Shorten your sessions to 30 seconds to avoid eye strain. Take a short screen break before trying again.")
        if asym_risk >= 15:
            recs.append("Practice using only the 'Horizontal' or 'Vertical' modes to strengthen the specific direction that feels slightly weaker.")

        return {
            "conclusion": conclusion,
            "stats": stats,
            "diseases": diseases,
            "recommendations": recs
        }