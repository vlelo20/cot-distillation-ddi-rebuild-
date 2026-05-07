"""
src/step12_local_generate.py

Local fallback generator for the 2x2 pilot when Narval is queue-stuck.

Uses transformers + bitsandbytes 4-bit quantization so a 4B model fits a
6 GB consumer GPU (e.g. RTX 4050 laptop). No vLLM dependency.

Reads pilot_prompts.jsonl produced by step12_pilot_2x2.py --prepare,
writes pilot_traces.jsonl in the SAME schema the existing --score mode
expects, so results from Narval and local can be scored identically (and
combined if you want apples-to-apples on a subset).

Resume-safe: skips (pair_uid, condition) combos already in the traces file.

Usage:
    python scripts/step12_local_generate.py \
        --data-dir ~/ddiproject/processed_v2 \
        --out-dir  ~/ddiproject/pilot \
        --model    Qwen/Qwen3-4B \
        --quant    nf4 \
        --max-new-tokens 1024 \
        [--limit 100]   # smoke-test on first N prompts
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# We only import the system prompt from teacher_prompt_v3; nothing else here
# depends on it, so it must already be on disk alongside this script.
from teacher_prompt_v3 import TEACHER_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _load_model(model_name: str, quant: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    log.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log.info(f"Loading model with {quant} quantization (this can take a few min on first run)...")
    if quant == "nf4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb,
            torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
    elif quant == "int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb,
            torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
    elif quant == "none":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown quant '{quant}' (expected nf4/int8/none)")
    model.eval()
    return tokenizer, model


def _build_chat_input(tokenizer, system_prompt: str, user_prompt: str,
                     enable_thinking: bool):
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    # Some chat templates accept enable_thinking; others ignore unknown kwargs.
    # If the model is Qwen3 (not 3.5) it has no thinking mode and the kwarg is
    # silently ignored.
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir",  type=Path, required=True)
    ap.add_argument("--model",    default="Qwen/Qwen3-4B")
    ap.add_argument("--quant",    default="nf4", choices=["nf4", "int8", "none"])
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature",    type=float, default=0.0)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--limit",    type=int, default=None,
                    help="Only generate first N pending prompts (smoke testing).")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Default OFF — Qwen3.5 family will skip <think> blocks.")
    args = ap.parse_args()

    prompts_path = args.out_dir / "pilot_prompts.jsonl"
    traces_path  = args.out_dir / "pilot_traces.jsonl"
    if not prompts_path.exists():
        raise SystemExit(f"Missing {prompts_path}")

    # Resume: skip (uid, cond) already done.
    done = set()
    if traces_path.exists():
        with open(traces_path) as f:
            for line in f:
                rec = json.loads(line)
                done.add((rec["pair_uid"], rec["condition"]))
        log.info(f"Resuming: {len(done):,} traces already in output")

    pending = []
    with open(prompts_path) as f:
        for line in f:
            rec = json.loads(line)
            if (rec["pair_uid"], rec["condition"]) in done:
                continue
            pending.append(rec)
    if args.limit:
        pending = pending[: args.limit]
        log.info(f"--limit set: only generating first {len(pending)} pending prompts")
    if not pending:
        log.info("Nothing to do.")
        return
    log.info(f"Pending: {len(pending):,} prompts")

    import torch
    torch.manual_seed(args.seed)

    tokenizer, model = _load_model(args.model, args.quant)

    # Open output append-mode and flush after each row so a Ctrl-C
    # never loses more than the current generation.
    with open(traces_path, "a", buffering=1) as fout:
        t_total = time.time()
        for i, rec in enumerate(pending, 1):
            chat_text = _build_chat_input(
                tokenizer, TEACHER_SYSTEM_PROMPT, rec["prompt_text"],
                enable_thinking=args.enable_thinking,
            )
            inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
            n_in = int(inputs["input_ids"].shape[1])

            t0 = time.time()
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=(args.temperature > 0),
                    pad_token_id=tokenizer.pad_token_id,
                )
                if args.temperature > 0:
                    gen_kwargs["temperature"] = args.temperature
                gen_ids = model.generate(**inputs, **gen_kwargs)
            elapsed = time.time() - t0

            # Slice off the prompt to keep just the generated text.
            new_ids = gen_ids[0, n_in:]
            trace = tokenizer.decode(new_ids, skip_special_tokens=True)
            n_out = int(new_ids.shape[0])

            scrubbed = {k: v for k, v in rec.items() if k != "prompt_text"}
            scrubbed.update({
                "trace": trace,
                "prompt_tokens": n_in,
                "generated_tokens": n_out,
                "finish_reason": "length" if n_out >= args.max_new_tokens else "stop",
                "_local_seconds": round(elapsed, 2),
            })
            fout.write(json.dumps(scrubbed) + "\n")

            if i == 1 or i % 25 == 0 or i == len(pending):
                rate = i / (time.time() - t_total)
                eta_min = (len(pending) - i) / rate / 60 if rate > 0 else 0
                log.info(
                    f"  {i:>4}/{len(pending)}  in={n_in} out={n_out} "
                    f"t={elapsed:.1f}s  rate={rate:.2f}/s  eta={eta_min:.1f} min"
                )

    log.info(f"Done. Total: {(time.time() - t_total)/60:.1f} min "
             f"-> {traces_path}")


if __name__ == "__main__":
    main()
