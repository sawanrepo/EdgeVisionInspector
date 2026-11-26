"""Command-line interface for EdgeVision Inspector workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from edgevision.config import collect_runtime_environment, get_logger, load_yaml
from edgevision.data.annotations import build_class_lookup, load_records_jsonl, parse_dataset
from edgevision.data.audit import run_data_audit
from edgevision.data.dataset import build_dataloader
from edgevision.data.splits import save_split_manifests, stratified_group_split
from edgevision.data.transforms import Preprocessor
from edgevision.evaluation.error_analysis import (
    analyze_detection_errors,
    save_error_overlays,
    write_failure_analysis_markdown,
)
from edgevision.evaluation.evaluate import (
    check_validation_regression,
    create_validation_baseline,
    evaluate_golden_test_once,
    select_operating_points_from_validation,
)
from edgevision.evaluation.metrics import compute_coco_map
from edgevision.evaluation.robustness import (
    evaluate_corruption_suite,
    evaluate_natural_hard_subset,
    save_robustness_report,
)
from edgevision.inference.benchmark import run_cpu_benchmark
from edgevision.inference.predict import OnnxPredictor
from edgevision.models.build import build_detector
from edgevision.optimization.export_onnx import export_adapter_to_onnx
from edgevision.optimization.quantize import build_calibration_records, quantize_onnx_static_qdq
from edgevision.optimization.validate_parity import validate_onnx_parity_batch
from edgevision.training.checkpoints import load_checkpoint
from edgevision.training.train import collect_predictions, train_detector

logger = get_logger(__name__)


def _cmd_env_check(args: argparse.Namespace) -> None:
    env = collect_runtime_environment()
    gpu_smoke: list[dict[str, Any]] = []
    if torch.cuda.is_available() and not args.cpu_only:
        for idx in range(torch.cuda.device_count()):
            dev = torch.device(f"cuda:{idx}")
            x = torch.randn((1, 3, 320, 320), device=dev)
            conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).to(dev)
            y = conv(x)
            torch.cuda.synchronize(dev)
            gpu_smoke.append({"device": f"cuda:{idx}", "output_shape": list(y.shape), "ok": True})
    env["gpu_smoke_tests"] = gpu_smoke
    out_text = json.dumps(env, indent=2, ensure_ascii=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out_text, encoding="utf-8")
    print(out_text)


def _cmd_audit(args: argparse.Namespace) -> None:
    ds_cfg = load_yaml(args.dataset_cfg)
    raw_dir = args.raw_dir or ds_cfg["paths"]["raw_dir"]
    parse_res = parse_dataset(raw_dir, ds_cfg)
    summary = run_data_audit(
        parse_res,
        dataset_cfg=ds_cfg,
        report_path=args.report_path,
        figures_dir=args.figures_dir,
    )
    print(json.dumps({"num_valid_images": summary["num_valid_images"], "total_instances": summary["total_instances"]}, indent=2))


def _cmd_split(args: argparse.Namespace) -> None:
    ds_cfg = load_yaml(args.dataset_cfg)
    raw_dir = args.raw_dir or ds_cfg["paths"]["raw_dir"]
    manifests_dir = args.manifests_dir or ds_cfg["paths"]["manifests_dir"]
    parse_res = parse_dataset(raw_dir, ds_cfg)
    split_cfg = ds_cfg.get("split", {})
    from edgevision.data.audit import analyze_duplicates

    analyze_duplicates(
        parse_res.records,
        phash_max_dist=int(split_cfg.get("near_duplicate_phash_max_dist", 4)),
        cosine_min=float(split_cfg.get("near_duplicate_cosine_min", 0.995)),
    )
    split_res = stratified_group_split(
        records=parse_res.records,
        num_classes=len(ds_cfg["classes"]),
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        seed=int(split_cfg.get("seed", 42)),
    )
    paths = save_split_manifests(split_res, manifests_dir=manifests_dir, dataset_cfg=ds_cfg)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


def _cmd_train(args: argparse.Namespace) -> None:
    input_size = tuple(args.input_size) if args.input_size else None
    summary = train_detector(
        model_cfg_path=args.model_cfg,
        dataset_cfg_path=args.dataset_cfg,
        preprocess_cfg_path=args.preprocess_cfg,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        output_dir=args.output_dir,
        device_str=args.device,
        epochs_override=args.epochs,
        input_size_override=input_size,
        resume_from=args.resume_from,
        use_class_weights=args.use_class_weights,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def _collect_onnx_predictions(
    predictor: OnnxPredictor,
    manifest_path: str | Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    records = load_records_jsonl(manifest_path)
    preds_by_img: dict[str, list[dict[str, Any]]] = {}
    gts_by_img: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        res = predictor.predict(rec.image_path, conf_threshold=0.05)
        preds_by_img[rec.image_id] = [
            {
                "class_id": d.class_id,
                "score": d.score,
                "bbox_xyxy": [d.xmin, d.ymin, d.xmax, d.ymax],
            }
            for d in res.detections
        ]
        gts_by_img[rec.image_id] = [
            {"class_id": b.class_id, "bbox_xyxy": [b.xmin, b.ymin, b.xmax, b.ymax]}
            for b in rec.boxes
        ]
    return preds_by_img, gts_by_img


def _cmd_evaluate_val(args: argparse.Namespace) -> None:
    ds_cfg = load_yaml(args.dataset_cfg)
    bench_cfg = load_yaml(args.benchmark_cfg).get("benchmark", {})
    specs, _ = build_class_lookup(ds_cfg)
    class_names = [s.name for s in specs]

    if args.onnx_path:
        predictor = OnnxPredictor(
            onnx_path=args.onnx_path,
            architecture=args.architecture,
            input_size=args.input_size,
            dataset_cfg=args.dataset_cfg,
            preprocess_cfg=args.preprocess_cfg,
            intra_op_num_threads=int(bench_cfg.get("intra_op_num_threads", 4)),
            inter_op_num_threads=int(bench_cfg.get("inter_op_num_threads", 1)),
            conf_threshold=0.05,
        )
        preds_by_img, gts_by_img = _collect_onnx_predictions(predictor, args.val_manifest)
        arch = args.architecture
        input_size = list(args.input_size)
    else:
        model_cfg = load_yaml(args.model_cfg)
        arch = str(model_cfg["model"]["architecture"])
        input_size = list(args.input_size or model_cfg["model"]["input_size"])
        device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        adapter = build_detector(model_cfg, override_pretrained=False, override_input_size=input_size).to(device)
        load_checkpoint(args.checkpoint, adapter, map_location=device)
        val_loader = build_dataloader(
            manifest_path=args.val_manifest,
            architecture=arch,
            input_size=input_size,
            batch_size=8,
            shuffle=False,
            preprocess_cfg=args.preprocess_cfg,
        )
        preds_by_img, gts_by_img = collect_predictions(adapter, val_loader, class_names, device=device)

    op_points = select_operating_points_from_validation(
        val_preds_by_image=preds_by_img,
        val_gts_by_image=gts_by_img,
        class_names=class_names,
        iou_threshold=float(bench_cfg.get("iou_threshold", 0.50)),
        target_macro_recall=float(bench_cfg.get("target_macro_recall", 0.90)),
        min_conf=float(bench_cfg.get("operating_point_min_conf", 0.05)),
        max_conf=float(bench_cfg.get("operating_point_max_conf", 0.95)),
        num_steps=int(bench_cfg.get("operating_point_sweep_steps", 91)),
        artifact_label=args.artifact_label,
    )
    op_dest = Path(args.operating_points_out)
    op_dest.parent.mkdir(parents=True, exist_ok=True)
    op_dest.write_text(json.dumps(op_points, indent=2, ensure_ascii=True), encoding="utf-8")

    baseline = create_validation_baseline(
        val_preds_by_image=preds_by_img,
        val_gts_by_image=gts_by_img,
        operating_points=op_points,
        class_names=class_names,
        val_manifest_path=args.val_manifest,
        model_info={"architecture": arch, "input_size": input_size, "artifact_label": args.artifact_label},
        bootstrap_replicates=int(bench_cfg.get("bootstrap_replicates", 2000)),
        bootstrap_seed=int(bench_cfg.get("bootstrap_seed", 42)),
        p95_cpu_latency_ms=args.p95_cpu_latency_ms,
    )
    base_dest = Path(args.baseline_out)
    base_dest.parent.mkdir(parents=True, exist_ok=True)
    base_dest.write_text(json.dumps(baseline, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"operating_points": str(op_dest), "baseline": str(base_dest)}, indent=2))


def _cmd_check_regression(args: argparse.Namespace) -> None:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    ds_cfg = load_yaml(args.dataset_cfg)
    specs, _ = build_class_lookup(ds_cfg)
    class_names = [s.name for s in specs]

    if args.onnx_path:
        predictor = OnnxPredictor(
            onnx_path=args.onnx_path,
            architecture=args.architecture,
            input_size=args.input_size,
            dataset_cfg=args.dataset_cfg,
            preprocess_cfg=args.preprocess_cfg,
            conf_threshold=0.05,
        )
        preds_by_img, gts_by_img = _collect_onnx_predictions(predictor, args.val_manifest)
    else:
        model_cfg = load_yaml(args.model_cfg)
        device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        adapter = build_detector(model_cfg, override_pretrained=False).to(device)
        load_checkpoint(args.checkpoint, adapter, map_location=device)
        loader = build_dataloader(
            args.val_manifest,
            architecture=model_cfg["model"]["architecture"],
            input_size=model_cfg["model"]["input_size"],
            batch_size=8,
            preprocess_cfg=args.preprocess_cfg,
        )
        preds_by_img, gts_by_img = collect_predictions(adapter, loader, class_names, device=device)

    op_points = select_operating_points_from_validation(preds_by_img, gts_by_img, class_names)
    recall_op = op_points["recall_target_operating_point"]
    op_metrics = (
        recall_op["validation_metrics"]
        if recall_op["feasible"]
        else op_points["quality_operating_point"]["validation_metrics"]
    )
    coco_val = compute_coco_map(preds_by_img, gts_by_img, num_classes=len(class_names))
    result = check_validation_regression(
        candidate_val_coco=coco_val,
        candidate_op_metrics=op_metrics,
        baseline=baseline,
        candidate_p95_cpu_latency_ms=args.p95_cpu_latency_ms,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["passed"]:
        raise SystemExit(1)


def _cmd_export_onnx(args: argparse.Namespace) -> None:
    model_cfg = load_yaml(args.model_cfg)
    input_size = tuple(args.input_size) if args.input_size else tuple(model_cfg["model"]["input_size"])
    adapter = build_detector(model_cfg, override_pretrained=False, override_input_size=input_size)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, adapter, map_location="cpu")
    opset = int(args.opset or model_cfg.get("onnx", {}).get("opset_version", 17))
    meta = export_adapter_to_onnx(adapter, args.output_onnx, opset_version=opset)

    if meta["status"] == "success" and args.val_manifest:
        ds_cfg = load_yaml(args.dataset_cfg)
        specs, _ = build_class_lookup(ds_cfg)
        class_names = [s.name for s in specs]
        val_records = load_records_jsonl(args.val_manifest)[: args.parity_samples]
        pre = Preprocessor(adapter.architecture, input_size=input_size, preprocess_cfg=args.preprocess_cfg)
        parity = validate_onnx_parity_batch(
            adapter=adapter,
            onnx_path=args.output_onnx,
            preprocessor=pre,
            image_sources=[r.image_path for r in val_records],
            class_names=class_names,
            output_report_path=args.parity_report,
        )
        meta["parity_report"] = parity
    print(json.dumps(meta, indent=2, ensure_ascii=True))


def _cmd_quantize_int8(args: argparse.Namespace) -> None:
    int8_cfg = load_yaml(args.int8_cfg)
    qcfg = int8_cfg["quantization"]
    calib_records = build_calibration_records(
        manifest_paths=args.calibration_manifests,
        num_images=int(qcfg.get("calibration_num_images", 200)),
        seed=int(qcfg.get("calibration_seed", 42)),
    )
    pre = Preprocessor(args.architecture, input_size=args.input_size, preprocess_cfg=args.preprocess_cfg)
    patterns = int8_cfg["fallback_ladder"][1].get("fp32_node_name_patterns") if len(int8_cfg.get("fallback_ladder", [])) > 1 else None
    meta = quantize_onnx_static_qdq(
        fp32_onnx_path=args.fp32_onnx,
        output_int8_path=args.output_int8,
        calibration_records=calib_records,
        preprocessor=pre,
        per_channel=bool(qcfg.get("per_channel", True)),
        keep_head_fp32=args.keep_head_fp32,
        fp32_node_patterns=patterns,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=True))


def _cmd_benchmark_cpu(args: argparse.Namespace) -> None:
    records = load_records_jsonl(args.manifest)
    if "golden" in Path(args.manifest).name.lower() or "test" in Path(args.manifest).name.lower():
        raise ValueError("Use validation or non-test images for CPU latency benchmarking.")
    report = run_cpu_benchmark(
        onnx_path=args.onnx_path,
        architecture=args.architecture,
        input_size=args.input_size,
        image_sources=[r.image_path for r in records[: args.num_unique_images]],
        benchmark_cfg=args.benchmark_cfg,
        dataset_cfg=args.dataset_cfg,
        preprocess_cfg=args.preprocess_cfg,
        conf_threshold=args.conf_threshold,
        operating_point_label=args.operating_point_label,
        output_report_path=args.output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


def _cmd_evaluate_golden(args: argparse.Namespace) -> None:
    ds_cfg = load_yaml(args.dataset_cfg)
    bench_cfg = load_yaml(args.benchmark_cfg).get("benchmark", {})
    specs, _ = build_class_lookup(ds_cfg)
    class_names = [s.name for s in specs]
    op_points = json.loads(Path(args.operating_points).read_text(encoding="utf-8"))

    predictor = OnnxPredictor(
        onnx_path=args.onnx_path,
        architecture=args.architecture,
        input_size=args.input_size,
        dataset_cfg=args.dataset_cfg,
        preprocess_cfg=args.preprocess_cfg,
        intra_op_num_threads=int(bench_cfg.get("intra_op_num_threads", 4)),
        inter_op_num_threads=int(bench_cfg.get("inter_op_num_threads", 1)),
        conf_threshold=0.05,
    )
    preds_by_img, gts_by_img = _collect_onnx_predictions(predictor, args.golden_manifest)
    evaluate_golden_test_once(
        preds_by_image=preds_by_img,
        gts_by_image=gts_by_img,
        operating_points=op_points,
        class_names=class_names,
        golden_manifest_path=args.golden_manifest,
        output_report_path=args.output_report,
        uncertainty_report_path=args.uncertainty_report,
        bootstrap_replicates=int(bench_cfg.get("bootstrap_replicates", 2000)),
        bootstrap_seed=int(bench_cfg.get("bootstrap_seed", 42)),
        allow_overwrite=args.allow_overwrite,
    )

    qual_thr = float(op_points["quality_operating_point"]["threshold"])
    golden_records = load_records_jsonl(args.golden_manifest)
    err = analyze_detection_errors(golden_records, preds_by_img, class_names, conf_threshold=qual_thr)
    overlays = save_error_overlays(
        golden_records,
        preds_by_img,
        class_names,
        output_dir=args.figures_dir,
        image_ids=err["top_error_image_ids"][:6],
        conf_threshold=qual_thr,
    )
    write_failure_analysis_markdown(err, overlays, args.failure_analysis_out)

    if args.hard_manifest and args.corruption_manifest:
        hard_preds, hard_gts = _collect_onnx_predictions(predictor, args.hard_manifest)
        hard_rep = evaluate_natural_hard_subset(hard_preds, hard_gts, class_names, conf_threshold=qual_thr)
        corr_entries = [json.loads(line) for line in Path(args.corruption_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        corr_preds: dict[str, list[dict[str, Any]]] = {}
        for entry in corr_entries:
            res = predictor.predict(entry["image_path"], conf_threshold=0.05)
            corr_preds[entry["image_id"]] = [
                {"class_id": d.class_id, "score": d.score, "bbox_xyxy": [d.xmin, d.ymin, d.xmax, d.ymax]}
                for d in res.detections
            ]
        corr_rep = evaluate_corruption_suite(corr_entries, corr_preds, class_names, conf_threshold=qual_thr)
        save_robustness_report(hard_rep, corr_rep, args.robustness_out)

    print(json.dumps({"final_test_report": args.output_report}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgevision", description="EdgeVision Inspector CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_env = sub.add_parser("env-check", help="Capture hardware and library environment")
    p_env.add_argument("--cpu-only", action="store_true")
    p_env.add_argument("--output", default=None)
    p_env.set_defaults(func=_cmd_env_check)

    p_audit = sub.add_parser("audit", help="Audit dataset images and annotations")
    p_audit.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    p_audit.add_argument("--raw-dir", default=None)
    p_audit.add_argument("--report-path", default="reports/data_audit.md")
    p_audit.add_argument("--figures-dir", default="reports/figures")
    p_audit.set_defaults(func=_cmd_audit)

    p_split = sub.add_parser("split", help="Create leakage-checked stratified splits and manifests")
    p_split.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    p_split.add_argument("--raw-dir", default=None)
    p_split.add_argument("--manifests-dir", default=None)
    p_split.set_defaults(func=_cmd_split)

    p_train = sub.add_parser("train", help="Train a detector architecture")
    p_train.add_argument("--model-cfg", required=True)
    p_train.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    p_train.add_argument("--preprocess-cfg", default="configs/preprocess.yaml")
    p_train.add_argument("--train-manifest", default="data/manifests/train.jsonl")
    p_train.add_argument("--val-manifest", default="data/manifests/val.jsonl")
    p_train.add_argument("--output-dir", required=True)
    p_train.add_argument("--device", default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--input-size", type=int, nargs=2, default=None)
    p_train.add_argument("--resume-from", default=None)
    p_train.add_argument("--use-class-weights", action="store_true")
    p_train.set_defaults(func=_cmd_train)

    p_val = sub.add_parser("evaluate-val", help="Freeze operating points and validation baseline")
    p_val.add_argument("--model-cfg", default="configs/yolox_tiny.yaml")
    p_val.add_argument("--checkpoint", default=None)
    p_val.add_argument("--onnx-path", default=None)
    p_val.add_argument("--architecture", default="yolox_tiny")
    p_val.add_argument("--input-size", type=int, nargs=2, default=[320, 320])
    p_val.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    p_val.add_argument("--preprocess-cfg", default="configs/preprocess.yaml")
    p_val.add_argument("--benchmark-cfg", default="configs/benchmark.yaml")
    p_val.add_argument("--val-manifest", default="data/manifests/val.jsonl")
    p_val.add_argument("--artifact-label", default="pytorch_fp32")
    p_val.add_argument("--operating-points-out", default="eval/operating_points.json")
    p_val.add_argument("--baseline-out", default="eval/validation_baseline.json")
    p_val.add_argument("--p95-cpu-latency-ms", type=float, default=None)
    p_val.add_argument("--device", default=None)
    p_val.set_defaults(func=_cmd_evaluate_val)

    p_reg = sub.add_parser("check-regression", help="Verify candidate against validation baseline")
    p_reg.add_argument("--baseline", default="eval/validation_baseline.json")
    p_reg.add_argument("--model-cfg", default="configs/yolox_tiny.yaml")
    p_reg.add_argument("--checkpoint", default=None)
    p_reg.add_argument("--onnx-path", default=None)
    p_reg.add_argument("--architecture", default="yolox_tiny")
    p_reg.add_argument("--input-size", type=int, nargs=2, default=[320, 320])
    p_reg.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    p_reg.add_argument("--preprocess-cfg", default="configs/preprocess.yaml")
    p_reg.add_argument("--val-manifest", default="data/manifests/val.jsonl")
    p_reg.add_argument("--p95-cpu-latency-ms", type=float, default=None)
    p_reg.add_argument("--device", default=None)
    p_reg.set_defaults(func=_cmd_check_regression)

    p_onnx = sub.add_parser("export-onnx", help="Export batch-1 FP32 ONNX graph and check parity")
    p_onnx.add_argument("--model-cfg", required=True)
    p_onnx.add_argument("--checkpoint", default=None)
    p_onnx.add_argument("--output-onnx", required=True)
    p_onnx.add_argument("--input-size", type=int, nargs=2, default=None)
    p_onnx.add_argument("--opset", type=int, default=None)
    p_onnx.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    p_onnx.add_argument("--preprocess-cfg", default="configs/preprocess.yaml")
    p_onnx.add_argument("--val-manifest", default=None)
    p_onnx.add_argument("--parity-samples", type=int, default=16)
    p_onnx.add_argument("--parity-report", default="reports/onnx_parity.json")
    p_onnx.set_defaults(func=_cmd_export_onnx)

    p_int8 = sub.add_parser("quantize-int8", help="Static QDQ INT8 quantization with train/val calibration")
    p_int8.add_argument("--fp32-onnx", required=True)
    p_int8.add_argument("--output-int8", required=True)
    p_int8.add_argument("--architecture", required=True)
    p_int8.add_argument("--input-size", type=int, nargs=2, default=[320, 320])
    p_int8.add_argument("--int8-cfg", default="configs/int8.yaml")
    p_int8.add_argument("--preprocess-cfg", default="configs/preprocess.yaml")
    p_int8.add_argument("--calibration-manifests", nargs="+", default=["data/manifests/train.jsonl", "data/manifests/val.jsonl"])
    p_int8.add_argument("--keep-head-fp32", action="store_true")
    p_int8.set_defaults(func=_cmd_quantize_int8)

    p_bench = sub.add_parser("benchmark-cpu", help="Benchmark batch-1 CPU ONNX Runtime latency")
    p_bench.add_argument("--onnx-path", required=True)
    p_bench.add_argument("--architecture", required=True)
    p_bench.add_argument("--input-size", type=int, nargs=2, default=[320, 320])
    p_bench.add_argument("--manifest", default="data/manifests/val.jsonl")
    p_bench.add_argument("--num-unique-images", type=int, default=50)
    p_bench.add_argument("--benchmark-cfg", default="configs/benchmark.yaml")
    p_bench.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    p_bench.add_argument("--preprocess-cfg", default="configs/preprocess.yaml")
    p_bench.add_argument("--conf-threshold", type=float, default=0.25)
    p_bench.add_argument("--operating-point-label", default="default")
    p_bench.add_argument("--output", default="benchmarks/benchmark_report.json")
    p_bench.set_defaults(func=_cmd_benchmark_cpu)

    p_gold = sub.add_parser("evaluate-golden", help="Run single controlled final evaluation on golden test")
    p_gold.add_argument("--onnx-path", required=True)
    p_gold.add_argument("--architecture", required=True)
    p_gold.add_argument("--input-size", type=int, nargs=2, default=[320, 320])
    p_gold.add_argument("--operating-points", default="eval/operating_points.json")
    p_gold.add_argument("--golden-manifest", default="data/manifests/golden_test.jsonl")
    p_gold.add_argument("--hard-manifest", default="data/manifests/natural_hard_subset.jsonl")
    p_gold.add_argument("--corruption-manifest", default="data/manifests/corruption_suite.jsonl")
    p_gold.add_argument("--dataset-cfg", default="configs/dataset.yaml")
    p_gold.add_argument("--preprocess-cfg", default="configs/preprocess.yaml")
    p_gold.add_argument("--benchmark-cfg", default="configs/benchmark.yaml")
    p_gold.add_argument("--output-report", default="eval/final_test_report.json")
    p_gold.add_argument("--uncertainty-report", default="reports/recall_uncertainty.json")
    p_gold.add_argument("--robustness-out", default="reports/robustness_report.json")
    p_gold.add_argument("--failure-analysis-out", default="reports/failure_analysis.md")
    p_gold.add_argument("--figures-dir", default="reports/figures")
    p_gold.add_argument("--allow-overwrite", action="store_true")
    p_gold.set_defaults(func=_cmd_evaluate_golden)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
