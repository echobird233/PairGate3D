"""Development-only calibration and held-out test for comparator/adaptive gates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hceg_ablation_suite_20261013 import run_review_response as rr  # noqa: E402


DEV = (0, 2, 3, 4, 7, 11, 17, 20)
TEST = tuple(rr.SEQUENCES)
DETECTORS = tuple(rr.DETECTORS)
PARAMS = rr.matched_parameters()
CAL_SPLIT = "gatecaldev20261109"
TEST_SPLIT = "gatecalholdout20261109"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def metric_summary(results: dict[str, dict[str, Any]], detectors: tuple[str, ...]) -> dict[str, Any]:
    per_detector = {}
    all_hota, all_assa, all_ids, all_fpfns = [], [], [], []
    for detector in detectors:
        result = results[detector]
        seq = result["metrics"]["per_sequence"]
        per_detector[detector] = {
            "HOTA": float(np.mean([float(row["HOTA"]) for row in seq.values()])),
            "AssA": float(np.mean([float(row["AssA"]) for row in seq.values()])),
            "IDSW": int(sum(int(row["IDSW"]) for row in seq.values())),
            "FP_FN": int(sum(int(row["FP"]) + int(row["FN"]) for row in seq.values())),
        }
        all_hota.append(per_detector[detector]["HOTA"])
        all_assa.append(per_detector[detector]["AssA"])
        all_ids.append(per_detector[detector]["IDSW"])
        all_fpfns.append(per_detector[detector]["FP_FN"])
    return {
        "per_detector": per_detector,
        "mean_sequence_equal_HOTA": float(np.mean(all_hota)),
        "mean_sequence_equal_AssA": float(np.mean(all_assa)),
        "total_IDSW": int(sum(all_ids)),
        "total_FP_plus_FN": int(sum(all_fpfns)),
    }


def run_one(phase: str, detector: str, family: str, name: str, parameters, *, sequences: tuple[int, ...], split: str, maha: float = rr.CHI2_3_95, adaptive=(0.0, 0.0)):
    return rr.run_job(
        phase, detector, family, parameters,
        sequences=sequences, split=split, output_name=name, gate_family=family,
        maha_threshold=maha, adaptive_coefficients=adaptive,
    )


def select_candidates(candidates: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    evaluated = []
    for candidate in candidates:
        results = candidate["results"]
        summary = metric_summary(results, DETECTORS)
        deltas = {
            d: summary["per_detector"][d]["HOTA"] - baseline["per_detector"][d]["HOTA"]
            for d in DETECTORS
        }
        candidate["development_metrics"] = summary
        candidate["per_detector_delta_HOTA"] = deltas
        candidate["guard_pass"] = min(deltas.values()) >= -0.5
        evaluated.append(candidate)
    eligible = [item for item in evaluated if item["guard_pass"]]
    pool = eligible if eligible else evaluated
    pool.sort(key=lambda x: (
        x["development_metrics"]["mean_sequence_equal_HOTA"],
        x["development_metrics"]["mean_sequence_equal_AssA"],
        -x["development_metrics"]["total_IDSW"],
        -x["development_metrics"]["total_FP_plus_FN"],
        -float(x.get("complexity", 0.0)),
    ), reverse=True)
    return {"selected": pool[0], "guard_passed_any": bool(eligible), "ranked": evaluated}


def concise_choice(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item["name"],
        "family": item["family"],
        "parameters": item["parameters"].__dict__,
        "maha_threshold": item["maha"],
        "adaptive_coefficients": list(item["adaptive"]),
        "complexity": item["complexity"],
        "development_metrics": item["development_metrics"],
        "per_detector_delta_HOTA": item["per_detector_delta_HOTA"],
        "guard_pass": item["guard_pass"],
    }


def comparator_specs():
    specs = []
    for value in (2.0, 2.5, 3.0, 3.223, 4.0, 5.0):
        specs.append({"family": "distance_only", "name": f"distance_d{str(value).replace('.', 'p')}", "parameters": rr.GateParameters(value, PARAMS.min_iou, 0, 1), "maha": rr.CHI2_3_95, "adaptive": (0.0, 0.0), "complexity": value})
    for value in (0.02, 0.05, 0.10, 0.1782, 0.25, 0.35):
        specs.append({"family": "overlap_only", "name": f"overlap_t{str(value).replace('.', 'p')}", "parameters": rr.GateParameters(PARAMS.max_distance, value, 0, 1), "maha": rr.CHI2_3_95, "adaptive": (0.0, 0.0), "complexity": value})
    for d, t in ((2.0,.05),(2.5,.10),(3.0,.15),(3.223,.1782),(4.0,.20),(5.0,.25)):
        specs.append({"family": "disjunction", "name": f"disj_d{str(d).replace('.', 'p')}_t{str(t).replace('.', 'p')}", "parameters": rr.GateParameters(d,t,0,1), "maha": rr.CHI2_3_95, "adaptive": (0.0,0.0), "complexity": d+t})
    for value in (2.366,4.642,6.251,7.815,9.348,11.345):
        specs.append({"family": "mahalanobis", "name": f"maha_q{str(value).replace('.', 'p')}", "parameters": PARAMS, "maha": value, "adaptive": (0.0,0.0), "complexity": value})
    return specs


def comparator_phase() -> dict[str, Any]:
    phase = "gate_calibration_dev"
    protected_before = rr.suite.protected_snapshot()
    baselines = {}
    for detector in DETECTORS:
        print(f"[gate calibration] dev baseline {detector}", flush=True)
        baselines[detector] = run_one(phase, detector, "no_hceg", "no_gate_dev", PARAMS, sequences=DEV, split=CAL_SPLIT)
    baseline = metric_summary(baselines, DETECTORS)
    candidates = []
    for spec in comparator_specs():
        results = {}
        for detector in DETECTORS:
            print(f"[gate calibration] {spec['name']} {detector}", flush=True)
            results[detector] = run_one(phase, detector, spec["family"], spec["name"], spec["parameters"], sequences=DEV, split=CAL_SPLIT, maha=spec["maha"], adaptive=spec["adaptive"])
        candidates.append({**spec, "results": results})

    families = ("distance_only", "overlap_only", "disjunction", "mahalanobis")
    selected = {}
    for family in families:
        selection = select_candidates([item for item in candidates if item["family"] == family], baseline)
        selected[family] = {"guard_passed_any": selection["guard_passed_any"], "selected": selection["selected"], "ranking": [
            {"name": item["name"], "development_metrics": item["development_metrics"], "per_detector_delta_HOTA": item["per_detector_delta_HOTA"], "guard_pass": item["guard_pass"], "parameters": item["parameters"].__dict__, "maha_threshold": item["maha"]}
            for item in sorted(selection["ranked"], key=lambda x: x["development_metrics"]["mean_sequence_equal_HOTA"], reverse=True)
        ]}

    test_results = {}
    test_baseline = {d: rr.source_result(d, "no_hceg") for d in DETECTORS}
    test_joint = {d: rr.source_result(d, "no_maturity") for d in DETECTORS}
    for family in families:
        spec = selected[family]["selected"]
        test_results[family] = {}
        for detector in DETECTORS:
            name = f"test_{spec['name']}"
            test_results[family][detector] = run_one("gate_calibration_holdout", detector, family, name, spec["parameters"], sequences=TEST, split=TEST_SPLIT, maha=spec["maha"], adaptive=spec["adaptive"])

    test_rows = []
    for family in families:
        for detector in DETECTORS:
            out = test_results[family][detector]["metrics"]["aggregate"]
            off = test_baseline[detector]["metrics"]["aggregate"]
            joint = test_joint[detector]["metrics"]["aggregate"]
            test_rows.append({"family":family,"detector":detector,"selected_dev_setting":selected[family]["selected"]["name"],"HOTA":out["HOTA"],"delta_HOTA_vs_no_gate":float(out["HOTA"])-float(off["HOTA"]),"delta_HOTA_vs_joint":float(out["HOTA"])-float(joint["HOTA"]),"AssA":out["AssA"],"MOTA":out["MOTA"],"FP":out["FP"],"FN":out["FN"],"IDSW":out["IDSW"]})
    protected_after = rr.suite.protected_snapshot()
    if protected_before != protected_after:
        raise RuntimeError("Protected source hashes changed during comparator calibration")
    summary = {
        "status":"completed","phase":"gate_calibration_and_holdout",
        "development_sequences":list(DEV),"holdout_sequences":list(TEST),"detectors":list(DETECTORS),
        "objective":"maximize equal-sequence HOTA across five paired detector streams; per-stream delta >= -0.5 guard; ties AssA, lower IDS, lower FP+FN, simpler setting",
        "fixed_hceg_thresholds_unchanged":PARAMS.__dict__,"development_baseline":baseline,
        "selected_by_family":{family:{"guard_passed_any":item["guard_passed_any"],"selected":concise_choice(item["selected"]),"ranking":item["ranking"]} for family,item in selected.items()},"holdout_rows":test_rows,
        "protected_before":protected_before,"protected_after":protected_after,
    }
    write_json(rr.OUT_ROOT/"gate_calibration_summary.json",summary)
    return summary


def adaptive_specs():
    specs=[]
    for a in (0.5,1.0,2.0):
        specs.append({"family":"adaptive_distance","name":f"adist_a{str(a).replace('.', 'p')}","parameters":PARAMS,"adaptive":(a,0.0),"maha":rr.CHI2_3_95,"complexity":a})
    for b in (0.5,1.0,2.0):
        specs.append({"family":"adaptive_overlap","name":f"aover_b{str(b).replace('.', 'p')}","parameters":PARAMS,"adaptive":(0.0,b),"maha":rr.CHI2_3_95,"complexity":b})
    for a in (0.5,1.0,2.0):
        for b in (0.5,1.0,2.0):
            specs.append({"family":"adaptive_joint","name":f"ajoint_a{str(a).replace('.', 'p')}_b{str(b).replace('.', 'p')}","parameters":PARAMS,"adaptive":(a,b),"maha":rr.CHI2_3_95,"complexity":a+b})
    return specs


def adaptive_phase() -> dict[str, Any]:
    phase="adaptive_gate_calibration_dev"
    protected_before=rr.suite.protected_snapshot()
    fixed={}
    no_gate={}
    for d in DETECTORS:
        print(f"[adaptive calibration] fixed baseline {d}",flush=True)
        fixed[d]=run_one(phase,d,"no_maturity","fixed_joint_dev",PARAMS,sequences=DEV,split=CAL_SPLIT)
        no_gate[d]=run_one(phase,d,"no_hceg","no_gate_dev",PARAMS,sequences=DEV,split=CAL_SPLIT)
    fixed_sum=metric_summary(fixed,DETECTORS)
    candidates=[]
    for spec in adaptive_specs():
        results={}
        for d in DETECTORS:
            print(f"[adaptive calibration] {spec['name']} {d}",flush=True)
            results[d]=run_one(phase,d,spec["family"],spec["name"],spec["parameters"],sequences=DEV,split=CAL_SPLIT,maha=spec["maha"],adaptive=spec["adaptive"])
        candidates.append({**spec,"results":results})
    selected={}
    for family in ("adaptive_distance","adaptive_overlap","adaptive_joint"):
        selection=select_candidates([x for x in candidates if x["family"]==family],fixed_sum)
        selected[family]={"guard_passed_any":selection["guard_passed_any"],"selected":selection["selected"],"ranking":[
            {"name":x["name"],"development_metrics":x["development_metrics"],"per_detector_delta_HOTA":x["per_detector_delta_HOTA"],"guard_pass":x["guard_pass"],"parameters":x["parameters"].__dict__,"adaptive":x["adaptive"]}
            for x in sorted(selection["ranked"],key=lambda x:x["development_metrics"]["mean_sequence_equal_HOTA"],reverse=True)
        ]}
    holdout={}
    test_fixed={d:rr.source_result(d,"no_maturity") for d in DETECTORS}
    for family in selected:
        spec=selected[family]["selected"]
        holdout[family]={}
        for d in DETECTORS:
            holdout[family][d]=run_one("adaptive_gate_holdout",d,family,f"test_{spec['name']}",spec["parameters"],sequences=TEST,split=TEST_SPLIT,maha=spec["maha"],adaptive=spec["adaptive"])
    rows=[]
    for family in selected:
        for d in DETECTORS:
            o=holdout[family][d]["metrics"]["aggregate"]; f=test_fixed[d]["metrics"]["aggregate"]
            rows.append({"family":family,"detector":d,"selected_dev_setting":selected[family]["selected"]["name"],"HOTA":o["HOTA"],"delta_HOTA_vs_fixed":float(o["HOTA"])-float(f["HOTA"]),"AssA":o["AssA"],"delta_AssA_vs_fixed":float(o["AssA"])-float(f["AssA"]),"MOTA":o["MOTA"],"FP":o["FP"],"FN":o["FN"],"IDSW":o["IDSW"],"delta_IDSW_vs_fixed":int(o["IDSW"])-int(f["IDSW"])})
    protected_after=rr.suite.protected_snapshot()
    if protected_before!=protected_after: raise RuntimeError("Protected source hashes changed during adaptive calibration")
    summary={"status":"completed","phase":"adaptive_calibration_and_holdout","development_sequences":list(DEV),"holdout_sequences":list(TEST),"covariance_definition":"sigma=sqrt(trace(P_xyz + R I)/3); predicted-box scale=sqrt(length*width)","distance_rule":"clip(3.223 + alpha*sigma, 2, 6) m","overlap_rule":"clip(0.1782*exp(-beta*sigma/sqrt(length*width)), 0.02, 0.1782)","coefficients_tested":{"alpha":[0.5,1.0,2.0],"beta":[0.5,1.0,2.0]},"development_fixed_gate":fixed_sum,"selected_by_family":{family:{"guard_passed_any":item["guard_passed_any"],"selected":concise_choice(item["selected"]),"ranking":item["ranking"]} for family,item in selected.items()},"holdout_rows":rows,"protected_before":protected_before,"protected_after":protected_after}
    write_json(rr.OUT_ROOT/"adaptive_gate_summary.json",summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("comparators","adaptive","both"),default="both")
    args=p.parse_args()
    started=time.time(); output={}
    if args.phase in ("comparators","both"):
        output["comparators"]=comparator_phase()
    if args.phase in ("adaptive","both"):
        output["adaptive"]=adaptive_phase()
    print(json.dumps({"status":"completed","phases":list(output),"elapsed_seconds":time.time()-started},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
