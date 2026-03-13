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
    Neosmith AI — RL-trained model (gpt-oss-120b, GRPO).

    Uses Tinker SDK for inference. Checkpoint configurable via NEOSMITH_CHECKPOINT env var.
    Default: codereview-rl-tinker 400-step checkpoint trained with YAML format reward.
    """

    # Configurable via NEOSMITH_CHECKPOINT env var
    SAMPLER_PATH = os.environ.get(
        "NEOSMITH_CHECKPOINT",
        "tinker://e279bf03-4d98-5371-b86a-30d6d20d1aff"
        ":train:0/sampler_weights/v1-rl-step250-codereview"
    )

    BASE_MODEL = "openai/gpt-oss-120b"

    def __init__(self):
        print("  [Neosmith] __init__ starting...")
        try:
            import tinker
            from tinker import types as tinker_types
            from tinker_cookbook import model_info, renderers
            from tinker_cookbook.tokenizer_utils import get_tokenizer
        except ImportError as e:
            print(f"  [Neosmith] IMPORT ERROR: {e}")
            raise

        api_key = os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise ValueError("TINKER_API_KEY is not set")
        print(f"  [Neosmith] TINKER_API_KEY present (length={len(api_key)})")

        self._types           = tinker_types
        self._renderers       = renderers
        try:
            self._sampling_client = tinker.ServiceClient(api_key=api_key) \
                                          .create_sampling_client(model_path=self.SAMPLER_PATH)

            print(f"  [Neosmith] Sampling client created OK (model_path={self.SAMPLER_PATH})")
            print(f"  [Neosmith] Sampling client created OK")
        except Exception as e:
            print(f"  [Neosmith] ERROR creating sampling client: {type(e).__name__}: {e}")
            raise

        try:
            tokenizer     = get_tokenizer(self.BASE_MODEL)
            renderer_name = model_info.get_recommended_renderer_name(self.BASE_MODEL) or "gpt_oss_system"
            self._renderer = renderers.get_renderer(renderer_name, tokenizer)
            print(f"  [Neosmith] Renderer created OK (renderer={renderer_name})")
        except Exception as e:
            print(f"  [Neosmith] ERROR creating renderer: {type(e).__name__}: {e}")
            raise
        print("  [Neosmith] __init__ complete")

        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self._last_response = None   # store last raw response for fallback

    @property
    def deployment_id(self):
        return None

    def _blocking_sample(self, model_input, temperature: float) -> str:
        """Synchronous Tinker call — runs in a thread pool."""
        # model_input is a ModelInput with multiple chunks
        if hasattr(model_input, 'chunks'):
            input_len = sum(len(c.tokens) for c in model_input.chunks)
        else:
            input_len = 0
        print(f"  [Neosmith] _blocking_sample called, input_tokens={input_len}")
        try:
            params = self._types.SamplingParams(
                max_tokens=8192,
                temperature=temperature,
                stop=self._renderer.get_stop_sequences(),
            )
            result = self._sampling_client.sample(
                prompt=model_input, num_samples=1, sampling_params=params
            ).result()
            print(f"  [Neosmith] Tinker SDK returned, output_tokens={len(result.sequences[0].tokens)}")
        except Exception as e:
            print(f"  [Neosmith] ERROR in Tinker SDK call: {type(e).__name__}: {e}")
            raise

        out_tokens = result.sequences[0].tokens
        self.total_input_tokens  += input_len
        self.total_output_tokens += len(out_tokens)

        # Try renderer parse first
        text = None
        try:
            parsed, _ = self._renderer.parse_response(out_tokens)
            text = self._renderers.get_text_content(parsed)
        except Exception as e:
            print(f"  [Neosmith] Renderer parse failed: {e}")

        # Fallback to raw token decode
        if not text:
            from tinker_cookbook.tokenizer_utils import get_tokenizer
            tokenizer = get_tokenizer(self.BASE_MODEL)
            text = tokenizer.decode(out_tokens)
            # Strip special tokens from raw decode
            for prefix in ("<|channel|>final<|message|>", "<|channel|>", "<|message|>"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
            print(f"  [Neosmith] Used raw decode fallback ({len(text)} chars)")

        # Strip markdown yaml fences if model wraps output
        if text:
            stripped = text.strip()
            if stripped.startswith("```yaml"):
                stripped = stripped[7:]
            elif stripped.startswith("```"):
                stripped = stripped[3:]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            text = stripped.strip()

        self._last_response = text
        print(f"  [Neosmith] Final response length={len(text) if text else 0}")
        return text

    async def chat_completion(self, model, system, user, temperature=0.2, img_path=None):
        print(f"  [Neosmith] chat_completion called (model param={model}, ignored — using Tinker SDK)")
        # Renderer is gpt_oss_no_sysprompt — does NOT support system messages.
        # Merge system prompt into the user message.
        combined_user = f"{system}\n\n{user}" if system else user
        messages    = [{"role": "user", "content": combined_user}]
        model_input = self._renderer.build_generation_prompt(messages)

        loop = asyncio.get_event_loop()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            text = await loop.run_in_executor(
                pool, self._blocking_sample, model_input, temperature
            )
        print(f"  [Neosmith] chat_completion done, response_length={len(text) if text else 0}")
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
        except Exception as e:
            print(f"  [CapturingReviewer] run() raised: {type(e).__name__}: {e}")
        finally:
            get_settings().config.publish_output = original

        # 1. Formatted markdown from _prepare_pr_review (if YAML parsed OK)
        if self.captured_review:
            return self.captured_review

        # 2. Stored artifact
        data = getattr(get_settings(), "data", None)
        if isinstance(data, dict) and data.get("artifact"):
            return data["artifact"]

        # 3. Raw AI prediction (Neosmith returns markdown, not YAML)
        raw = getattr(self, "prediction", None)
        if raw:
            print(f"  [CapturingReviewer] Using raw prediction ({len(raw)} chars)")
            return raw

        # 4. Check ai_handler for last response
        handler = getattr(self, "ai_handler", None)
        if handler and hasattr(handler, "_last_response") and handler._last_response:
            return handler._last_response

        return "(No review generated)"


class CapturingImprover(PRCodeSuggestions):
    async def get_suggestions(self):
        original = get_settings().config.publish_output
        get_settings().config.publish_output = False
        try:
            await self.run()
        except Exception as e:
            print(f"  [CapturingImprover] run() raised: {type(e).__name__}: {e}")
        finally:
            get_settings().config.publish_output = original

        # 1. Stored artifact
        data = getattr(get_settings(), "data", None)
        if isinstance(data, dict) and data.get("artifact"):
            return data["artifact"]

        # 2. Raw AI prediction (Neosmith returns markdown, not YAML)
        raw = getattr(self, "prediction", None)
        if raw:
            print(f"  [CapturingImprover] Using raw prediction ({len(raw)} chars)")
            return raw

        # 3. Check prediction_list for extended mode
        plist = getattr(self, "prediction_list", None)
        if plist:
            return "\n\n---\n\n".join(str(p) for p in plist if p)

        return "(No suggestions generated)"


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
    print(f"  neosmith : {neo_model} (Tinker SDK)")
    print(f"  endpoint : {NeosmithHandler.SAMPLER_PATH}")
    print(f"  gpt-5.2  : {openai_model}")

    # ── Run all 4 tasks (Neosmith first) ─────────────────────────────────────
    print(f"\n[1/4] Neosmith  /review  ...")
    neo_review,  neo_rv_ms,  neo_rv_in,  neo_rv_out  = await run_review(pr_url, NeosmithHandler)
    print(f"      {neo_rv_ms}ms  |  {neo_rv_in:,} in / {neo_rv_out:,} out tokens")

    print(f"[2/4] Neosmith  /improve ...")
    neo_improve, neo_im_ms,  neo_im_in,  neo_im_out  = await run_improve(pr_url, NeosmithHandler)
    print(f"      {neo_im_ms}ms  |  {neo_im_in:,} in / {neo_im_out:,} out tokens")

    print(f"\n[3/4] GPT-5.2   /review  ...")
    gpt_review,  gpt_rv_ms,  gpt_rv_in,  gpt_rv_out  = await run_review(pr_url, GPTHandler)
    print(f"      {gpt_rv_ms}ms  |  {gpt_rv_in:,} in / {gpt_rv_out:,} out tokens")

    print(f"[4/4] GPT-5.2   /improve ...")
    gpt_improve, gpt_im_ms,  gpt_im_in,  gpt_im_out  = await run_improve(pr_url, GPTHandler)
    print(f"      {gpt_im_ms}ms  |  {gpt_im_in:,} in / {gpt_im_out:,} out tokens")

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

    def _winner_emoji(neo_val, gpt_val, lower_is_better=True):
        """Return emoji indicating who won this metric."""
        if lower_is_better:
            return "🏆" if neo_val <= gpt_val else ""
        else:
            return "🏆" if neo_val >= gpt_val else ""

    # ── Scorecard: count wins ──────────────────────────────────────────────────
    neo_wins = 0
    gpt_wins = 0
    # Latency wins (lower is better)
    if neo_rv_ms <= gpt_rv_ms: neo_wins += 1
    else: gpt_wins += 1
    if neo_im_ms <= gpt_im_ms: neo_wins += 1
    else: gpt_wins += 1
    # Cost wins (lower is better)
    if neo_total_cost <= gpt_total_cost: neo_wins += 1
    else: gpt_wins += 1
    # Review length — longer review = more thorough (higher is better)
    neo_rv_len = len(str(neo_review or ""))
    gpt_rv_len = len(str(gpt_review or ""))
    if neo_rv_len >= gpt_rv_len: neo_wins += 1
    else: gpt_wins += 1
    # Improve length — more suggestions = more thorough
    neo_im_len = len(str(neo_improve or ""))
    gpt_im_len = len(str(gpt_improve or ""))
    if neo_im_len >= gpt_im_len: neo_wins += 1
    else: gpt_wins += 1

    verdict = "🏆 **Neosmith wins overall**" if neo_wins >= gpt_wins else "GPT-5.2 leads"

    # ── Latency table ─────────────────────────────────────────────────────────
    latency_table = (
        f"| Tool | Neosmith | GPT-5.2 | Winner |\n"
        f"|---|---|---|---|\n"
        f"| `/review`  | **{neo_rv_ms}ms** | {gpt_rv_ms}ms | {_speedup(gpt_rv_ms, neo_rv_ms)} |\n"
        f"| `/improve` | **{neo_im_ms}ms** | {gpt_im_ms}ms | {_speedup(gpt_im_ms, neo_im_ms)} |\n"
        f"| **Total**  | **{neo_total_ms}ms** | {gpt_total_ms}ms | {_speedup(gpt_total_ms, neo_total_ms)} |"
    )

    # ── Cost table (Neosmith column first) ─────────────────────────────────────
    cost_table = (
        f"| | Neosmith | GPT-5.2 | Advantage |\n"
        f"|---|---|---|---|\n"
        f"| Input price | **\\$1.250/1M** | \\$1.750/1M | Neosmith **28% cheaper** |\n"
        f"| Output price | **\\$5.000/1M** | \\$14.000/1M | Neosmith **64% cheaper** |\n"
        f"| `/review` tokens | **{neo_rv_in:,} in / {neo_rv_out:,} out** | {gpt_rv_in:,} in / {gpt_rv_out:,} out | |\n"
        f"| `/improve` tokens | **{neo_im_in:,} in / {neo_im_out:,} out** | {gpt_im_in:,} in / {gpt_im_out:,} out | |\n"
        f"| **`/review` cost** | **\\${neo_rv_cost:.4f}** | \\${gpt_rv_cost:.4f} | |\n"
        f"| **`/improve` cost** | **\\${neo_im_cost:.4f}** | \\${gpt_im_cost:.4f} | |\n"
        f"| **Total cost** | **\\${neo_total_cost:.4f}** | \\${gpt_total_cost:.4f} | **Save \\${savings:.4f} ({savings_pct:.0f}%)** |"
    )

    # ── Quality table ──────────────────────────────────────────────────────────
    quality_table = (
        f"| Metric | Neosmith | GPT-5.2 | |\n"
        f"|---|---|---|---|\n"
        f"| Review length | **{neo_rv_len:,} chars** | {gpt_rv_len:,} chars | {_winner_emoji(gpt_rv_len, neo_rv_len, lower_is_better=True)} |\n"
        f"| Suggestions length | **{neo_im_len:,} chars** | {gpt_im_len:,} chars | {_winner_emoji(gpt_im_len, neo_im_len, lower_is_better=True)} |"
    )

    # ── Diffs ─────────────────────────────────────────────────────────────────
    review_diff  = _diff_block(gpt_review,  neo_review,
                               f"gpt/{openai_model}", f"neosmith/{neo_model}")
    improve_diff = _diff_block(gpt_improve, neo_improve,
                               f"gpt/{openai_model}", f"neosmith/{neo_model}")

    # ── GitHub comment (Neosmith-first layout) ────────────────────────────────
    comment = f"""\
## 🚀 Neosmith AI vs GPT-5.2 — PR Review Comparison

> **Tinker endpoint**: `{NeosmithHandler.SAMPLER_PATH}`

| | Model | Type | Scorecard |
|---|---|---|---|
| 🚀 **Neosmith** | `{neo_model}` | RL-trained · GRPO | **{neo_wins}/5 wins** |
| 🤖 GPT-5.2 | `{openai_model}` | Standard OpenAI | {gpt_wins}/5 wins |

### {verdict}

---

## 🚀 Neosmith Review

{neo_review}

---

## 🚀 Neosmith Code Suggestions

{neo_improve}

---

### ⚡ Latency Comparison

{latency_table}

---

### 💰 Cost Comparison

{cost_table}

---

### 📊 Quality Comparison

{quality_table}

---

<details>
<summary>🤖 GPT-5.2 Review (for reference)</summary>

{gpt_review}
</details>

<details>
<summary>🤖 GPT-5.2 Code Suggestions (for reference)</summary>

{gpt_improve}
</details>

<details>
<summary>Review diff — Neosmith vs GPT-5.2</summary>

{review_diff}
</details>

<details>
<summary>Improve diff — Neosmith vs GPT-5.2</summary>

{improve_diff}
</details>
"""

    print(f"\n  ── Results ──")
    print(f"  Verdict       : {verdict}")
    print(f"  Scorecard     : Neosmith {neo_wins}/5 | GPT-5.2 {gpt_wins}/5")
    print(f"  Neosmith cost : ${neo_total_cost:.4f}")
    print(f"  GPT-5.2  cost : ${gpt_total_cost:.4f}")
    print(f"  Savings       : ${savings:.4f} ({savings_pct:.0f}%)")
    print(f"  Tinker endpoint: {NeosmithHandler.SAMPLER_PATH}")

    print("\nPosting comparison comment to PR ...")
    from pr_agent.git_providers import get_git_provider_with_context
    get_git_provider_with_context(pr_url).publish_comment(comment)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
