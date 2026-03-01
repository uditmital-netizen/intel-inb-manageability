#!/usr/bin/env python3
"""
Compare PR review + improve: GPT-5.2 vs Neosmith AI (RL-trained).
Runs /review and /improve for both models, shows latency, token cost,
and savings — posts a single comparison comment to the GitHub PR.

Usage:
    python compare_reviews.py                          # reads everything from .env
    python compare_reviews.py --pr_url <url>           # override PR
    python compare_reviews.py --openai_model gpt-4o    # override model
"""

import asyncio
import argparse
import difflib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import litellm
from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.config_loader import get_settings
from pr_agent.tools.pr_reviewer import PRReviewer
from pr_agent.tools.pr_code_suggestions import PRCodeSuggestions

# ── .env auto-discovery ───────────────────────────────────────────────────────
_DEFAULT_ENV = (
    Path(__file__).parent   # pr-agent/
    .parent                 # workspace/workspace/
    .parent                 # workspace/
    / "rl_optimization/dev/rl_tinker/.env"
)


def _load_env(env_file=None):
    from dotenv import load_dotenv
    path = Path(env_file) if env_file else _DEFAULT_ENV
    if path.exists():
        load_dotenv(path, override=False)
        print(f"  env  : loaded from {path}")
    else:
        print(f"  env  : {path} not found — using shell environment")


# ── Pricing constants (per 1M tokens) ────────────────────────────────────────
GPT_INPUT_PRICE  = 1.750   # $/1M input tokens
GPT_OUTPUT_PRICE = 14.000  # $/1M output tokens

NEO_INPUT_PRICE  = 1.250   # $/1M input tokens
NEO_OUTPUT_PRICE = 5.000   # $/1M output tokens


def calc_cost(input_tokens: int, output_tokens: int,
              input_price: float, output_price: float) -> float:
    return (input_tokens / 1_000_000) * input_price \
         + (output_tokens / 1_000_000) * output_price


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GPT Handler  (wraps LiteLLMAIHandler, adds token tracking)
# ─────────────────────────────────────────────────────────────────────────────

class GPTHandler(LiteLLMAIHandler):
    """
    LiteLLMAIHandler + token usage tracking.
    Uses litellm.token_counter so we don't need to parse the raw response object.
    """

    def __init__(self):
        super().__init__()
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    async def chat_completion(self, model, system, user, temperature=0.2, img_path=None):
        text, finish_reason = await super().chat_completion(
            model, system, user, temperature, img_path
        )
        try:
            messages = [{"role": "system", "content": system},
                        {"role": "user",   "content": user}]
            self.total_input_tokens  += litellm.token_counter(model=model, messages=messages)
            self.total_output_tokens += litellm.token_counter(model=model, text=text or "")
        except Exception:
            pass   # token counting is best-effort
        return text, finish_reason


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Neosmith AI Handler  (Tinker SDK — will swap to LiteLLM endpoint later)
# ─────────────────────────────────────────────────────────────────────────────

class NeosmithHandler(BaseAiHandler):
    """
    Neosmith AI — RL-trained model (gpt-oss-120b, GRPO step-200).

    Currently backed by the Tinker SDK directly.
    TODO: swap to a LiteLLM proxy endpoint once it's running:
        resp = await litellm.acompletion(model=f"openai/{self.model_name}",
                                         api_base=NEOSMITH_API_URL, ...)

    Checkpoint:
        tinker://ee92788e-a9ce-5c85-8612-df8d19f75f12:train:0/sampler_weights/final
    """

    SAMPLER_PATH = (
        "tinker://ee92788e-a9ce-5c85-8612-df8d19f75f12"
        ":train:0/sampler_weights/final"
    )
    BASE_MODEL = "openai/gpt-oss-120b"

    def __init__(self):
        import tinker
        from tinker import types as tinker_types
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        api_key = os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise ValueError("TINKER_API_KEY is not set")

        self._types           = tinker_types
        self._renderers       = renderers
        self._sampling_client = tinker.ServiceClient(api_key=api_key) \
                                      .create_sampling_client(model_path=self.SAMPLER_PATH)

        tokenizer     = get_tokenizer(self.BASE_MODEL)
        renderer_name = model_info.get_recommended_renderer_name(self.BASE_MODEL) or "gpt_oss_system"
        self._renderer = renderers.get_renderer(renderer_name, tokenizer)

        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    @property
    def deployment_id(self):
        return None

    def _blocking_sample(self, model_input, temperature: float) -> str:
        """Synchronous Tinker call — runs in a thread pool."""
        params = self._types.SamplingParams(
            max_tokens=2048,
            temperature=temperature,
            stop=self._renderer.get_stop_sequences(),
        )
        result = self._sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=params
        ).result()

        # model_input is a token-id sequence → its length = input token count
        self.total_input_tokens  += len(model_input)
        self.total_output_tokens += len(result.sequences[0].tokens)

        parsed, _ = self._renderer.parse_response(result.sequences[0].tokens)
        return self._renderers.get_text_content(parsed)

    async def chat_completion(self, model, system, user, temperature=0.2, img_path=None):
        messages    = [{"role": "system", "content": system},
                       {"role": "user",   "content": user}]
        model_input = self._renderer.build_generation_prompt(messages)

        loop = asyncio.get_event_loop()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            text = await loop.run_in_executor(
                pool, self._blocking_sample, model_input, temperature
            )
        return text, "stop"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Capturing tools  (suppress publish, capture output)
# ─────────────────────────────────────────────────────────────────────────────

class CapturingReviewer(PRReviewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_review = None

    def _prepare_pr_review(self):
        result = super()._prepare_pr_review()
        self.captured_review = result
        return result

    async def get_review(self):
        original = get_settings().config.publish_output
        get_settings().config.publish_output = False
        try:
            await self.run()
        finally:
            get_settings().config.publish_output = original
        return self.captured_review or "(No review generated)"


class CapturingImprover(PRCodeSuggestions):
    async def get_suggestions(self):
        original = get_settings().config.publish_output
        get_settings().config.publish_output = False
        try:
            await self.run()
        finally:
            get_settings().config.publish_output = original
        data = get_settings().get("data", {})
        return data.get("artifact", "(No suggestions generated)")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Timed runners  — return (text, ms, input_tokens, output_tokens)
# ─────────────────────────────────────────────────────────────────────────────

async def run_review(pr_url, handler_class):
  
    t0       = time.perf_counter()
    reviewer = CapturingReviewer(pr_url, ai_handler=handler_class)
    result   = await reviewer.get_review()
    ms       = round((time.perf_counter() - t0) * 1000)
    tok_in   = getattr(reviewer.ai_handler, "total_input_tokens",  0)
    tok_out  = getattr(reviewer.ai_handler, "total_output_tokens", 0)
    return result, ms, tok_in, tok_out


async def run_improve(pr_url, handler_class):
    t0       = time.perf_counter()
    improver = CapturingImprover(pr_url, ai_handler=handler_class)
    result   = await improver.get_suggestions()
    ms       = round((time.perf_counter() - t0) * 1000)
    tok_in   = getattr(improver.ai_handler, "total_input_tokens",  0)
    tok_out  = getattr(improver.ai_handler, "total_output_tokens", 0)
    return result, ms, tok_in, tok_out


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Diff helper
# ─────────────────────────────────────────────────────────────────────────────

def _diff_block(text_a, text_b, label_a, label_b):
    lines_a = [l + "\n" for l in text_a.splitlines()]
    lines_b = [l + "\n" for l in text_b.splitlines()]
    diff = list(difflib.unified_diff(
        lines_a, lines_b, fromfile=label_a, tofile=label_b, lineterm=""
    ))
    if not diff:
        return "_Both models produced identical output._"
    capped = diff[:80]
    if len(diff) > 80:
        capped.append(f"... ({len(diff) - 80} more lines truncated)\n")
    return "```diff\n" + "".join(capped) + "\n```"


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="GPT-5.2 vs Neosmith AI — PR review + improve comparison"
    )
    parser.add_argument("--pr_url",       default=None)
    parser.add_argument("--openai_model", default=None)
    parser.add_argument("--env_file",     default=None)
    args = parser.parse_args()

    _load_env(args.env_file)

    pr_url       = args.pr_url       or os.environ.get("PR_URL")
    openai_model = args.openai_model or os.environ.get("OPENAI_MODEL") or "gpt-5.2-2025-12-11"
    openai_key   = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    github_token = os.environ.get("GITHUB_TOKEN")
    neo_model    = os.environ.get("NEOSMITH_MODEL", "gpt-oss-120b")

    missing = [k for k, v in {
        "PR_URL":         pr_url,
        "OPENAI_API_KEY": openai_key,
        "GITHUB_TOKEN":   github_token,
    }.items() if not v]
    if missing:
        print(f"ERROR: Missing env values: {', '.join(missing)}")
        sys.exit(1)

    get_settings().set("CONFIG.GIT_PROVIDER", "github")
    get_settings().set("GITHUB.USER_TOKEN",   github_token)
    get_settings().set("OPENAI.KEY",          openai_key)
    get_settings().set("CONFIG.MODEL",        openai_model)

    print(f"\n  pr       : {pr_url}")
    print(f"  gpt-5.2  : {openai_model}")
    print(f"  neosmith : {neo_model} (Tinker SDK)")

    # ── Run all 4 tasks ───────────────────────────────────────────────────────
    print(f"\n[1/4] GPT-5.2   /review  ...")
    gpt_review,  gpt_rv_ms,  gpt_rv_in,  gpt_rv_out  = await run_review(pr_url, GPTHandler)
    print(f"      {gpt_rv_ms}ms  |  {gpt_rv_in:,} in / {gpt_rv_out:,} out tokens")

    print(f"[2/4] GPT-5.2   /improve ...")
    gpt_improve, gpt_im_ms,  gpt_im_in,  gpt_im_out  = await run_improve(pr_url, GPTHandler)
    print(f"      {gpt_im_ms}ms  |  {gpt_im_in:,} in / {gpt_im_out:,} out tokens")

    print(f"\n[3/4] Neosmith  /review  ...")
    neo_review,  neo_rv_ms,  neo_rv_in,  neo_rv_out  = await run_review(pr_url, NeosmithHandler)
    print(f"      {neo_rv_ms}ms  |  {neo_rv_in:,} in / {neo_rv_out:,} out tokens")

    print(f"[4/4] Neosmith  /improve ...")
    neo_improve, neo_im_ms,  neo_im_in,  neo_im_out  = await run_improve(pr_url, NeosmithHandler)
    print(f"      {neo_im_ms}ms  |  {neo_im_in:,} in / {neo_im_out:,} out tokens")

    # ── Aggregate totals ──────────────────────────────────────────────────────
    gpt_total_ms  = gpt_rv_ms  + gpt_im_ms
    neo_total_ms  = neo_rv_ms  + neo_im_ms
    gpt_total_in  = gpt_rv_in  + gpt_im_in
    gpt_total_out = gpt_rv_out + gpt_im_out
    neo_total_in  = neo_rv_in  + neo_im_in
    neo_total_out = neo_rv_out + neo_im_out

    # ── Costs ─────────────────────────────────────────────────────────────────
    gpt_rv_cost  = calc_cost(gpt_rv_in,  gpt_rv_out,  GPT_INPUT_PRICE, GPT_OUTPUT_PRICE)
    gpt_im_cost  = calc_cost(gpt_im_in,  gpt_im_out,  GPT_INPUT_PRICE, GPT_OUTPUT_PRICE)
    neo_rv_cost  = calc_cost(neo_rv_in,  neo_rv_out,  NEO_INPUT_PRICE, NEO_OUTPUT_PRICE)
    neo_im_cost  = calc_cost(neo_im_in,  neo_im_out,  NEO_INPUT_PRICE, NEO_OUTPUT_PRICE)
   
    gpt_total_cost = gpt_rv_cost + gpt_im_cost
    neo_total_cost = neo_rv_cost + neo_im_cost
    savings        = gpt_total_cost - neo_total_cost
    savings_pct    = (savings / gpt_total_cost * 100) if gpt_total_cost else 0

    def _speedup(a, b):
        if b and a > b:  return f"**{a/b:.1f}x faster** 🚀"
        if a and b > a:  return f"**{b/a:.1f}x faster** (GPT)"
        return "same"

    # ── Latency table ─────────────────────────────────────────────────────────
    latency_table = (
        f"| Tool | GPT-5.2 | Neosmith | Winner |\n"
        f"|---|---|---|---|\n"
        f"| `/review`  | {gpt_rv_ms}ms | {neo_rv_ms}ms | {_speedup(gpt_rv_ms, neo_rv_ms)} |\n"
        f"| `/improve` | {gpt_im_ms}ms | {neo_im_ms}ms | {_speedup(gpt_im_ms, neo_im_ms)} |\n"
        f"| **Total**  | **{gpt_total_ms}ms** | **{neo_total_ms}ms** | {_speedup(gpt_total_ms, neo_total_ms)} |"
    )

    # ── Cost table ────────────────────────────────────────────────────────────
    cost_table = (
        f"| | GPT-5.2 | Neosmith | Pricing |\n"
        f"|---|---|---|---|\n"
        f"| Input price | \\$1.750/1M | \\$1.250/1M | Neosmith **28% cheaper** |\n"
        f"| Output price | \\$14.000/1M | \\$5.000/1M | Neosmith **64% cheaper** |\n"
        f"| `/review` tokens | {gpt_rv_in:,} in / {gpt_rv_out:,} out | {neo_rv_in:,} in / {neo_rv_out:,} out | |\n"
        f"| `/improve` tokens | {gpt_im_in:,} in / {gpt_im_out:,} out | {neo_im_in:,} in / {neo_im_out:,} out | |\n"
        f"| **`/review` cost** | **\\${gpt_rv_cost:.4f}** | **\\${neo_rv_cost:.4f}** | |\n"
        f"| **`/improve` cost** | **\\${gpt_im_cost:.4f}** | **\\${neo_im_cost:.4f}** | |\n"
        f"| **Total cost** | **\\${gpt_total_cost:.4f}** | **\\${neo_total_cost:.4f}** | **Save \\${savings:.4f} ({savings_pct:.0f}%)** |"
    )

    # ── Diffs ─────────────────────────────────────────────────────────────────
    review_diff  = _diff_block(gpt_review,  neo_review,
                               f"gpt/{openai_model}", f"neosmith/{neo_model}")
    improve_diff = _diff_block(gpt_improve, neo_improve,
                               f"gpt/{openai_model}", f"neosmith/{neo_model}")

    # ── GitHub comment ────────────────────────────────────────────────────────
    comment = f"""\
## 🔬 PR Analysis: GPT-5.2 vs Neosmith AI

| | Model | Type |
|---|---|---|
| 🤖 **GPT-5.2** | `{openai_model}` | Standard OpenAI |
| 🚀 **Neosmith** | `{neo_model}` | RL-trained · GRPO · step-200 |

---

### ⚡ Latency

{latency_table}

---

### 💰 Token Cost & Savings

{cost_table}

---

## `/review` Results

### 🤖 GPT-5.2 Review

{gpt_review}

---

### 🚀 Neosmith Review

{neo_review}

<details>
<summary>Review diff — lines Neosmith changed vs GPT-5.2</summary>

{review_diff}
</details>

---

## `/improve` Results

### 🤖 GPT-5.2 Code Suggestions

{gpt_improve}

---

### 🚀 Neosmith Code Suggestions

{neo_improve}

<details>
<summary>Improve diff — lines Neosmith changed vs GPT-5.2</summary>

{improve_diff}
</details>
"""

    print(f"\n  GPT-5.2  cost : ${gpt_total_cost:.4f}")
    print(f"  Neosmith cost : ${neo_total_cost:.4f}")
    print(f"  Savings       : ${savings:.4f} ({savings_pct:.0f}%)")

    print("\nPosting comparison comment to PR ...")
    from pr_agent.git_providers import get_git_provider_with_context
    get_git_provider_with_context(pr_url).publish_comment(comment)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
