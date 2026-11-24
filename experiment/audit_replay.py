"""Compare the archived primary SQL outputs with the later independent reruns."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def inspect(folder, full):
    out = dict(files={}, cases=0, repeat_plan_executions=0, nonidentical_repeats=0,
               max_repeat_abs=0.0, max_focal_repeat_abs=0.0, max_vs_primary_abs=0.0,
               max_focal_vs_primary_abs=0.0, changed_top_classes=0)
    for meta in sorted(folder.glob("*.json")):
        d = json.loads(meta.read_text())
        if "cases" not in d: continue
        raw = meta.with_suffix(".npz")
        out["files"][meta.name]=hashlib.sha256(meta.read_bytes()).hexdigest()
        out["files"][raw.name]=hashlib.sha256(raw.read_bytes()).hexdigest()
        assert d["code"]["query.py"] == hashlib.sha256((HERE/"query.py").read_bytes()).hexdigest()
        with np.load(raw, allow_pickle=False) as current, np.load(HERE/"records"/raw.name, allow_pickle=False) as primary:
            for c in d["cases"]:
                out["cases"]+=1
                prefix=f"p{c['panel']}_{c['recipe']}_{c['mode']}_"
                assert np.array_equal(current[prefix+"ids"],primary[prefix+"ids"])
                for plan in ("separate","pooled"):
                    logs=c["repeat_check"][plan]
                    out["repeat_plan_executions"]+=d["repeats"]
                    out["nonidentical_repeats"]+=logs["nonidentical"]
                    out["max_repeat_abs"]=max(out["max_repeat_abs"],logs["max_abs"])
                    out["max_focal_repeat_abs"]=max(out["max_focal_repeat_abs"],logs["focal_max_abs"])
                    old,new=primary[prefix+plan],current[prefix+plan]
                    diff=np.abs(old-new)
                    out["max_vs_primary_abs"]=max(out["max_vs_primary_abs"],float(diff.max()))
                    out["max_focal_vs_primary_abs"]=max(out["max_focal_vs_primary_abs"],float(diff[:c["n"]].max()))
                    out["changed_top_classes"]+=int(np.count_nonzero(old.argmax(1)!=new.argmax(1)))
    assert out["cases"]==(80 if full else 4) and out["changed_top_classes"]==0
    assert out["max_vs_primary_abs"]<=1e-6 and out["max_repeat_abs"]<=1e-6
    return out

if __name__=="__main__":
    base=HERE/"replay"
    report={p.name:inspect(p,p.name=="full") for p in sorted(base.iterdir()) if p.is_dir()}
    assert set(report)=={"full","process-1","process-2","process-3","process-4","eighteen"}
    print(json.dumps(report,indent=2))
