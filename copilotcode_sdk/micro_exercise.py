"""LLM-based micro and chain exercises for copilotcode_sdk.

Each exercise runs a real CopilotCodeClient session, sends prompts,
and verifies results using a Haiku LLM verifier call.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Sequence

from .exercise import (
    ExerciseReport,
    SubsystemResult,
    build_exercise_config,
    _capture_ground_truth,
)

if TYPE_CHECKING:
    from .client import CopilotCodeClient, CopilotCodeSession


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class VerifyResult:
    """Result of an LLM verifier call."""
    passed: bool
    reasoning: str
    token_usage: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0


class ExerciseFailure(Exception):
    """Raised when the LLM verifier judges an exercise as failed."""
    def __init__(self, reasoning: str) -> None:
        self.reasoning = reasoning
        super().__init__(reasoning)


@dataclass
class _ExerciseDef:
    """Internal definition of a registered exercise."""
    name: str
    kind: str  # "micro" or "chain"
    func: Callable[["ExerciseRunner"], Coroutine[Any, Any, None]]
    timeout: float


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

MICRO_EXERCISES: list[_ExerciseDef] = []
CHAIN_EXERCISES: list[_ExerciseDef] = []


def micro_exercise(name: str, timeout: float = 300.0):
    """Decorator to register a micro-exercise."""
    def wrapper(func: Callable[["ExerciseRunner"], Coroutine[Any, Any, None]]):
        MICRO_EXERCISES.append(_ExerciseDef(name=name, kind="micro", func=func, timeout=timeout))
        return func
    return wrapper


def chain_exercise(name: str, timeout: float = 480.0):
    """Decorator to register a chain exercise."""
    def wrapper(func: Callable[["ExerciseRunner"], Coroutine[Any, Any, None]]):
        CHAIN_EXERCISES.append(_ExerciseDef(name=name, kind="chain", func=func, timeout=timeout))
        return func
    return wrapper


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

VERIFIER_PROMPT_TEMPLATE = """\
You are verifying whether an AI agent completed a task correctly.

## Task Prompt
{task_prompt}

## Agent Response
{agent_response}

## Tool Calls Made
{tool_calls}

## Ground Truth
{ground_truth}

## Rubric
{rubric}

Judge whether the agent's response satisfies the rubric. Be strict — if the rubric \
says a tool must have been called, check the Tool Calls section for evidence. If it \
says content must match, verify against Ground Truth.

Return ONLY a JSON object (no markdown fencing):
{{"passed": true, "reasoning": "one paragraph explaining why"}}
or
{{"passed": false, "reasoning": "one paragraph explaining what failed"}}
"""


def build_verifier_prompt(
    *,
    task_prompt: str,
    agent_response: str,
    tool_calls: str,
    ground_truth: str,
    rubric: str,
) -> str:
    return VERIFIER_PROMPT_TEMPLATE.format(
        task_prompt=task_prompt,
        agent_response=agent_response,
        tool_calls=tool_calls,
        ground_truth=ground_truth,
        rubric=rubric,
    )


async def _call_verifier(
    client: "CopilotCodeClient",
    prompt: str,
    timeout: float = 60.0,
) -> VerifyResult:
    """Send a verification prompt to a new session and parse the result."""
    import logging
    from .exercise import _extract_last_assistant_text

    session = await client.create_session(
        session_id=f"verifier-{uuid.uuid4().hex[:8]}",
    )
    await session.send_and_wait(prompt, timeout=timeout)
    messages = await session.get_messages()
    text = _extract_last_assistant_text(messages)

    # Capture verifier session cost
    v_cost = session.cumulative_cost.total if hasattr(session, 'cumulative_cost') else 0.0
    v_tokens = {
        "input": session.state.total_input_tokens if hasattr(session, 'state') else 0,
        "output": session.state.total_output_tokens if hasattr(session, 'state') else 0,
    }
    logger = logging.getLogger("copilotcode.exercise")
    logger.info(
        "VERIFIER cost=$%.6f in=%d out=%d",
        v_cost, v_tokens.get("input", 0), v_tokens.get("output", 0),
    )

    # Parse JSON from response
    try:
        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return VerifyResult(
                passed=bool(data.get("passed", False)),
                reasoning=str(data.get("reasoning", "No reasoning provided.")),
                token_usage=v_tokens,
                cost=v_cost,
            )
    except (json.JSONDecodeError, KeyError):
        pass

    # If we can't parse, treat as error
    return VerifyResult(
        passed=False,
        reasoning=f"Verifier response could not be parsed as JSON: {text[:500]}",
        token_usage=v_tokens,
        cost=v_cost,
    )


# ---------------------------------------------------------------------------
# ExerciseRunner
# ---------------------------------------------------------------------------


class ExerciseRunner:
    """Test harness for micro and chain exercises."""

    def __init__(
        self,
        client: "CopilotCodeClient",
        verifier_client: "CopilotCodeClient",
        temp_dir: Path,
        session: "CopilotCodeSession",
    ) -> None:
        self._client = client
        self._verifier_client = verifier_client
        self._temp_dir = temp_dir
        self._session = session
        self._last_prompt: str = ""
        self._last_response: str = ""
        self._last_tool_calls: list[dict[str, Any]] = []
        self._ground_truth: dict[str, Any] = {}

    @property
    def temp_dir(self) -> Path:
        return self._temp_dir

    def write_fixture(self, name: str, content: str) -> Path:
        """Write a fixture file into the temp directory."""
        path = self._temp_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def read_fixture(self, name: str) -> str:
        """Read a fixture file from the temp directory."""
        path = self._temp_dir / name
        return path.read_text(encoding="utf-8")

    def delete_fixture(self, name: str) -> None:
        """Delete a fixture file from the temp directory."""
        path = self._temp_dir / name
        if path.exists():
            path.unlink()

    def fixture_exists(self, name: str) -> bool:
        """Check if a fixture file exists."""
        return (self._temp_dir / name).exists()

    @property
    def memory_dir(self) -> Path | None:
        """The memory store's directory, if available."""
        store = getattr(self._client, "_memory_store", None)
        return store.memory_dir if store else None

    def list_memory_files(self) -> list[Path]:
        """List all .md files in the memory directory (excluding MEMORY.md)."""
        mdir = self.memory_dir
        if mdir is None or not mdir.exists():
            return []
        return [
            p for p in sorted(mdir.rglob("*.md"))
            if p.name != "MEMORY.md" and p.name != "session_memory.md"
        ]

    def read_memory_contents(self) -> str:
        """Read and concatenate all memory file contents for ground truth."""
        files = self.list_memory_files()
        if not files:
            return "<NO MEMORY FILES FOUND>"
        parts = []
        for f in files:
            parts.append(f"--- {f.name} ---\n{f.read_text(encoding='utf-8')}")
        return "\n\n".join(parts)

    async def create_new_session(self) -> None:
        """Replace the current session with a fresh one (same config/memory root)."""
        self._session = await self._client.create_session(
            session_id=f"recall-{uuid.uuid4().hex[:8]}",
        )
        self._last_prompt = ""
        self._last_response = ""
        self._last_tool_calls = []

    async def prompt(self, text: str, timeout: float = 120.0) -> str:
        """Send a prompt to the session and return the response text."""
        from .exercise import _extract_last_assistant_text

        self._last_prompt = text
        await self._session.send_and_wait(text, timeout=timeout)
        messages = await self._session.get_messages()

        # Extract response text
        self._last_response = _extract_last_assistant_text(messages)

        # Extract tool calls from messages
        self._last_tool_calls = _extract_tool_calls(messages)

        # Capture ground truth
        self._ground_truth = _capture_ground_truth(self._session)

        return self._last_response

    async def verify(
        self,
        response: str,
        rubric: str,
        ground_truth: dict[str, Any] | None = None,
    ) -> VerifyResult:
        """Verify the response using an LLM verifier call."""
        # Merge runner ground truth with any exercise-specific ground truth
        merged_gt = dict(self._ground_truth)
        if ground_truth:
            merged_gt.update(ground_truth)

        # Format tool calls for the verifier
        tool_calls_text = json.dumps(self._last_tool_calls, indent=2) if self._last_tool_calls else "No tool calls recorded."

        prompt = build_verifier_prompt(
            task_prompt=self._last_prompt,
            agent_response=response,
            tool_calls=tool_calls_text,
            ground_truth=json.dumps(merged_gt, indent=2, default=str),
            rubric=rubric,
        )

        result = await _call_verifier(self._verifier_client, prompt)

        if not result.passed:
            raise ExerciseFailure(result.reasoning)

        return result


def _extract_tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract tool call information from session messages."""
    tool_calls: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            to_dict = getattr(msg, "to_dict", None)
            if callable(to_dict):
                msg = to_dict()
            else:
                try:
                    msg = vars(msg)
                except TypeError:
                    continue

        # Check for tool_use blocks in content
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append({
                        "tool": block.get("name", "unknown"),
                        "arguments": block.get("input", {}),
                    })
                elif isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_calls.append({
                        "tool_result": block.get("tool_use_id", ""),
                        "content": str(block.get("content", ""))[:500],
                    })

        # Check SDK event format
        data = msg.get("data") or {}
        if isinstance(data, dict):
            tool_requests = data.get("tool_requests") or data.get("toolRequests") or []
            for req in tool_requests:
                if isinstance(req, dict):
                    tool_calls.append({
                        "tool": req.get("name", req.get("tool", "unknown")),
                        "arguments": req.get("input", req.get("arguments", {})),
                    })

    return tool_calls


# ---------------------------------------------------------------------------
# Exercise definitions — Micro
# ---------------------------------------------------------------------------

SALES_CSV = """\
product,units_sold,price_per_unit
Widget A,150,12.50
Widget B,80,25.00
Gadget X,200,8.75
Gadget Y,45,50.00
Tool Alpha,120,15.00
Tool Beta,60,30.00
Part M,300,5.00
Part N,90,22.50
Device Q,35,75.00
Device R,110,18.00\
"""

EMPLOYEES_CSV = """\
name,department,salary
Alice,Engineering,95000
Bob,Engineering,88000
Carol,Marketing,72000
Dave,Marketing,68000
Eve,Sales,82000
Frank,Sales,78000
Grace,Engineering,102000\
"""

DEPARTMENTS_CSV = """\
department,budget,location
Engineering,500000,Building A
Marketing,200000,Building B
Sales,300000,Building C\
"""


@micro_exercise(name="read_file")
async def _exercise_read_file(runner: ExerciseRunner) -> None:
    runner.write_fixture("data.txt", "Alice,30\nBob,25\nCarol,35\n")
    response = await runner.prompt("Read data.txt and tell me what's in it.")
    await runner.verify(
        response,
        rubric=(
            "A Read or file-read tool call was made targeting data.txt. "
            "Response references the actual data: Alice (age 30), Bob (age 25), "
            "and Carol (age 35). Content must not be hallucinated."
        ),
        ground_truth={"file_content": "Alice,30\nBob,25\nCarol,35\n"},
    )


@micro_exercise(name="write_file")
async def _exercise_write_file(runner: ExerciseRunner) -> None:
    response = await runner.prompt(
        "Create a file called output.txt containing exactly three lines: "
        "'alpha', 'beta', 'gamma' — one per line, no extra whitespace."
    )
    file_content = runner.read_fixture("output.txt") if runner.fixture_exists("output.txt") else "<FILE NOT FOUND>"
    await runner.verify(
        response,
        rubric=(
            "A Write or file-write tool call was made. output.txt exists in the "
            "working directory. File content is exactly three lines: alpha, beta, gamma."
        ),
        ground_truth={"output_file_content": file_content},
    )


@micro_exercise(name="shell_command")
async def _exercise_shell_command(runner: ExerciseRunner) -> None:
    runner.write_fixture("a.txt", "file a")
    runner.write_fixture("b.txt", "file b")
    response = await runner.prompt(
        "Run ls in the current directory and tell me what files are here."
    )
    await runner.verify(
        response,
        rubric=(
            "A Bash or shell tool call was made running ls or equivalent. "
            "Response mentions both a.txt and b.txt."
        ),
    )


@micro_exercise(name="multi_tool")
async def _exercise_multi_tool(runner: ExerciseRunner) -> None:
    runner.write_fixture("prices.csv", "item,price\nApple,2.50\nBanana,1.20\nCherry,4.00\n")
    runner.write_fixture("inventory.csv", "item,quantity\nApple,100\nBanana,250\nCherry,50\n")
    # Correct answer: Apple=250, Banana=300, Cherry=200 -> Banana highest
    response = await runner.prompt(
        "Read both prices.csv and inventory.csv, then tell me which item "
        "has the highest total value (price * quantity)."
    )
    await runner.verify(
        response,
        rubric=(
            "Two Read or file-read tool calls were made (one per file). "
            "Response identifies Banana as the item with highest total value "
            "(1.20 * 250 = 300). The calculation must be based on actual file "
            "data, not hallucinated."
        ),
        ground_truth={
            "correct_answer": "Banana",
            "correct_value": 300.0,
            "all_values": {"Apple": 250.0, "Banana": 300.0, "Cherry": 200.0},
        },
    )


@micro_exercise(name="error_recovery")
async def _exercise_error_recovery(runner: ExerciseRunner) -> None:
    runner.write_fixture("good.txt", "This file exists and has content.")
    response = await runner.prompt(
        "Try to read bad.txt, then read good.txt. Tell me what happened with each."
    )
    await runner.verify(
        response,
        rubric=(
            "Agent attempted to read bad.txt and encountered an error or failure. "
            "Agent then successfully read good.txt using a Read tool call. "
            "Response acknowledges the first file failed and reports the content "
            "of the second file accurately: 'This file exists and has content.'"
        ),
        ground_truth={"good_file_content": "This file exists and has content."},
    )


@micro_exercise(name="follow_instructions")
async def _exercise_follow_instructions(runner: ExerciseRunner) -> None:
    response = await runner.prompt(
        "Create a file called report.csv with a header row 'name,score,grade' "
        "followed by exactly 5 data rows. Use realistic student names and scores "
        "between 0-100. Assign grades: A for 90+, B for 80-89, C for 70-79, "
        "D for 60-69, F below 60."
    )
    file_content = runner.read_fixture("report.csv") if runner.fixture_exists("report.csv") else "<FILE NOT FOUND>"
    await runner.verify(
        response,
        rubric=(
            "A Write tool call was made. report.csv exists. File has exactly "
            "6 lines (1 header + 5 data). Header is 'name,score,grade'. Each "
            "data row has 3 comma-separated fields. Scores are between 0-100. "
            "Grades match the scoring rubric: A for 90+, B for 80-89, C for "
            "70-79, D for 60-69, F below 60."
        ),
        ground_truth={"report_file_content": file_content},
    )


# ---------------------------------------------------------------------------
# Exercise definitions — Chain
# ---------------------------------------------------------------------------


@chain_exercise(name="read_transform_report")
async def _exercise_read_transform_report(runner: ExerciseRunner) -> None:
    runner.write_fixture("sales.csv", SALES_CSV)

    # Pre-compute correct revenues
    correct_revenues = {
        "Widget A": 150 * 12.50,   # 1875.00
        "Widget B": 80 * 25.00,    # 2000.00
        "Gadget X": 200 * 8.75,    # 1750.00
        "Gadget Y": 45 * 50.00,    # 2250.00
        "Tool Alpha": 120 * 15.00, # 1800.00
        "Tool Beta": 60 * 30.00,   # 1800.00
        "Part M": 300 * 5.00,      # 1500.00
        "Part N": 90 * 22.50,      # 2025.00
        "Device Q": 35 * 75.00,    # 2625.00
        "Device R": 110 * 18.00,   # 1980.00
    }
    top_product = max(correct_revenues, key=correct_revenues.get)  # Device Q

    # Step 1: Read
    r1 = await runner.prompt("Read sales.csv and summarize the columns and row count.")
    await runner.verify(
        r1,
        rubric=(
            "A Read tool was called targeting sales.csv. Response correctly "
            "identifies the three columns (product, units_sold, price_per_unit) "
            "and states there are 10 data rows."
        ),
    )

    # Step 2: Transform
    r2 = await runner.prompt(
        "Calculate the total revenue (units_sold * price_per_unit) for each product. "
        "Write the results to summary.txt with one line per product in the format "
        "'product: $revenue'."
    )
    summary_content = runner.read_fixture("summary.txt") if runner.fixture_exists("summary.txt") else "<FILE NOT FOUND>"
    await runner.verify(
        r2,
        rubric=(
            "A Write tool was called. summary.txt exists. Each line follows the "
            "'product: $N' format. Revenue values must be correct based on the "
            "actual CSV data. For example, Widget A should be $1875.00 "
            "(150 * 12.50), Device Q should be $2625.00 (35 * 75.00)."
        ),
        ground_truth={
            "correct_revenues": correct_revenues,
            "summary_file_content": summary_content,
        },
    )

    # Step 3: Report
    r3 = await runner.prompt(
        "Read summary.txt and write a one-paragraph report identifying "
        "the top-selling product by revenue."
    )
    summary_after = runner.read_fixture("summary.txt") if runner.fixture_exists("summary.txt") else "<FILE NOT FOUND>"
    await runner.verify(
        r3,
        rubric=(
            "A Read tool was called targeting summary.txt. Response identifies "
            "Device Q as the top product by revenue ($2625.00). The answer must "
            "match the actual data in summary.txt, not be hallucinated."
        ),
        ground_truth={
            "summary_file_content": summary_after,
            "correct_top_product": top_product,
            "correct_top_revenue": correct_revenues[top_product],
        },
    )


@chain_exercise(name="multi_file_synthesis")
async def _exercise_multi_file_synthesis(runner: ExerciseRunner) -> None:
    runner.write_fixture("employees.csv", EMPLOYEES_CSV)
    runner.write_fixture("departments.csv", DEPARTMENTS_CSV)

    # Pre-compute correct answers
    # Engineering avg: (95000+88000+102000)/3 = 95000
    # Marketing avg: (72000+68000)/2 = 70000
    # Sales avg: (82000+78000)/2 = 80000
    correct_dept = "Engineering"
    correct_avg = 95000.0

    # Step 1: Read
    r1 = await runner.prompt(
        "Read both employees.csv and departments.csv. Tell me how many "
        "employees and departments there are."
    )
    await runner.verify(
        r1,
        rubric=(
            "Two Read tool calls were made (one per file). Response states "
            "there are 7 employees and 3 departments."
        ),
        ground_truth={"employee_count": 7, "department_count": 3},
    )

    # Step 2: Merge
    r2 = await runner.prompt(
        "Create merged.csv that joins employees with their department info. "
        "Columns: name,department,salary,budget,location. Write one row per employee."
    )
    merged_content = runner.read_fixture("merged.csv") if runner.fixture_exists("merged.csv") else "<FILE NOT FOUND>"
    await runner.verify(
        r2,
        rubric=(
            "A Write tool was called. merged.csv exists with header "
            "'name,department,salary,budget,location'. There are 7 data rows "
            "(one per employee). Each employee's budget and location match their "
            "department in departments.csv. For example, Alice in Engineering "
            "should have budget 500000 and location Building A."
        ),
        ground_truth={"merged_file_content": merged_content},
    )

    # Step 3: Analyze
    r3 = await runner.prompt(
        "Read merged.csv and tell me which department has the highest "
        "average salary. Include the average salary amount."
    )
    merged_after = runner.read_fixture("merged.csv") if runner.fixture_exists("merged.csv") else "<FILE NOT FOUND>"
    await runner.verify(
        r3,
        rubric=(
            "A Read tool was called targeting merged.csv. Response identifies "
            "Engineering as the department with the highest average salary. "
            "The average salary amount should be $95,000 (or within $1 of it). "
            "This must be based on actual data, not hallucinated."
        ),
        ground_truth={
            "merged_file_content": merged_after,
            "correct_department": correct_dept,
            "correct_average_salary": correct_avg,
        },
    )


# ---------------------------------------------------------------------------
# Exercise definitions — Memory lifecycle
# ---------------------------------------------------------------------------

SALES_Q1_CSV = """\
region,product,revenue,units
Northeast,Widget A,4200,84
Northeast,Widget B,3100,62
Southeast,Widget A,5800,116
Southeast,Widget B,890,18
Midwest,Widget A,3600,72
Midwest,Widget B,2200,44
West,Widget A,6100,122
West,Widget B,1500,30\
"""

SALES_Q2_CSV = """\
region,product,revenue,units
Northeast,Widget A,4800,96
Northeast,Widget B,3500,70
Southeast,Widget A,6200,124
Southeast,Widget B,1400,28
Midwest,Widget A,3100,62
Midwest,Widget B,1900,38
West,Widget A,5900,118
West,Widget B,1600,32\
"""

# Pre-computed ground truth
_Q1_REGION_TOTALS = {
    "Northeast": 7300, "Southeast": 6690, "Midwest": 5800, "West": 7600,
}
_Q2_REGION_TOTALS = {
    "Northeast": 8300, "Southeast": 7600, "Midwest": 5000, "West": 7500,
}
_Q1_VS_Q2_DELTAS = {
    "Northeast": 1000, "Southeast": 910, "Midwest": -800, "West": -100,
}


@chain_exercise(name="memory_lifecycle")
async def _exercise_memory_lifecycle(runner: ExerciseRunner) -> None:
    runner.write_fixture("sales_q1.csv", SALES_Q1_CSV)

    # Step 1 — Analyze Q1 and save findings
    r1 = await runner.prompt(
        "Read sales_q1.csv and analyze the data. Identify the top region by "
        "total revenue and any products with revenue below $1000.\n\n"
        "Two stakeholder questions came up during this analysis:\n"
        "1. The VP of Sales asked: is the Southeast region viable long-term "
        "or should we consider reallocating its budget?\n"
        "2. Marketing wants to know: are Widget B sales declining across all "
        "regions or just specific ones?\n\n"
        "Answer both questions based on the data. Then save your key findings, "
        "analytical conclusions, and these open stakeholder questions to memory "
        "so future sessions can build on this work."
    )
    await runner.verify(
        r1,
        rubric=(
            "Agent read sales_q1.csv. Identified West as the top region by "
            "revenue ($7,600). Identified Southeast Widget B ($890) as below "
            "$1,000. Answered both stakeholder questions with data-backed "
            "reasoning. Called a memory-save tool or wrote a memory file to "
            "persist findings. All findings must be based on actual CSV data."
        ),
        ground_truth={
            "q1_region_totals": _Q1_REGION_TOTALS,
            "q1_top_region": "West",
            "q1_top_revenue": 7600,
            "q1_low_revenue_product": "Southeast Widget B ($890)",
        },
    )

    # Step 2 — Verify memory content
    r2 = await runner.prompt(
        "What findings have been saved to memory about this project? List them."
    )
    memory_contents = runner.read_memory_contents()
    await runner.verify(
        r2,
        rubric=(
            "Agent accessed its memory system and reported findings that include: "
            "(a) West as the top region by revenue, (b) Southeast Widget B as "
            "the low-revenue outlier, (c) the VP's question about Southeast "
            "viability, (d) Marketing's question about Widget B trends. The "
            "memory must contain derived analytical conclusions and stakeholder "
            "questions — NOT a copy of raw CSV data. If the memory files contain "
            "a dump of the CSV rows, this FAILS."
        ),
        ground_truth={
            "memory_file_contents": memory_contents,
            "expected_facts": [
                "West is top region (~$7,600)",
                "Southeast Widget B is low-revenue outlier ($890)",
                "VP question: Southeast viability / budget reallocation",
                "Marketing question: Widget B decline patterns",
            ],
        },
    )

    # Step 3 — Trigger extraction and remove Q1 data
    r3 = await runner.prompt(
        "Read sales_q1.csv again and calculate the average revenue per region. "
        "Also run `echo extraction_trigger_1`, `echo extraction_trigger_2`, "
        "and `echo extraction_trigger_3`."
    )
    await runner.verify(
        r3,
        rubric="Agent performed the file read and shell commands.",
    )

    # Post-step: delete Q1 data so session 2 cannot access it
    runner.delete_fixture("sales_q1.csv")

    # Step 4 — New session recalls and extends
    await runner.create_new_session()
    runner.write_fixture("sales_q2.csv", SALES_Q2_CSV)

    r4 = await runner.prompt(
        "Analyze sales_q2.csv. You've analyzed this data before in a previous "
        "session — check your memories for prior findings. Compare Q2 results "
        "to what you found in Q1. Which regions improved? Which declined? "
        "Also follow up on the open questions from the last analysis session."
    )
    memory_contents_after = runner.read_memory_contents()
    await runner.verify(
        r4,
        rubric=(
            "Agent referenced prior Q1 findings FROM MEMORY — specifically "
            "West/$7,600 as the top Q1 region and Southeast Widget B/$890 as "
            "the outlier. These MUST come from memory recall, not hallucination "
            "(the Q1 CSV file has been deleted and Q1 data is not in the prompt). "
            "Agent correctly analyzed Q2 data from the file. Comparison identifies: "
            "Northeast grew (+$1,000), Southeast grew (+$910), Midwest declined "
            "(-$800), West declined (-$100). Agent followed up on BOTH stakeholder "
            "questions — the VP's question about Southeast viability (now answerable "
            "with Q2 growth data showing Southeast improved) and Marketing's "
            "question about Widget B trends (now answerable with cross-region Q2 "
            "data) — WITHOUT the questions being re-stated in the step 4 prompt. "
            "If the agent does not mention or follow up on the stakeholder "
            "questions, this FAILS."
        ),
        ground_truth={
            "q1_region_totals": _Q1_REGION_TOTALS,
            "q2_region_totals": _Q2_REGION_TOTALS,
            "q1_vs_q2_deltas": _Q1_VS_Q2_DELTAS,
            "q1_top_region": "West",
            "q2_top_region": "Northeast",
            "memory_file_contents": memory_contents_after,
            "stakeholder_questions": [
                "VP: Southeast viability / budget reallocation",
                "Marketing: Widget B decline patterns across regions",
            ],
            "anti_cheat_note": (
                "Q1 CSV was deleted before this step. If the agent's Q1 facts "
                "are correct, they came from memory. If incorrect or vague, "
                "memory recall is broken."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Runner orchestration
# ---------------------------------------------------------------------------


async def _run_single_micro_exercise(
    exercise: _ExerciseDef,
    client: "CopilotCodeClient",
    verifier_client: "CopilotCodeClient",
) -> SubsystemResult:
    """Run a single exercise and return a SubsystemResult."""
    start = time.monotonic()
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"copilotcode_exercise_{exercise.name}_"))

        # Create a session pointed at the temp dir
        from .client import CopilotCodeClient
        from dataclasses import replace as dc_replace
        exercise_cfg = dc_replace(client.config, working_directory=str(temp_dir))
        exercise_client = CopilotCodeClient(exercise_cfg)

        session = await exercise_client.create_session(
            session_id=f"micro-{exercise.name}-{uuid.uuid4().hex[:8]}",
        )

        runner = ExerciseRunner(
            client=exercise_client,
            verifier_client=verifier_client,
            temp_dir=temp_dir,
            session=session,
        )

        await exercise.func(runner)

        elapsed = time.monotonic() - start
        # Aggregate cost: exercise session + verifier sessions
        session_cost = session.cumulative_cost.total if hasattr(session, 'cumulative_cost') else 0.0
        total_tokens = {
            "input": session.state.total_input_tokens if hasattr(session, 'state') else 0,
            "output": session.state.total_output_tokens if hasattr(session, 'state') else 0,
        }
        import logging
        logging.getLogger("copilotcode.exercise").info(
            "EXERCISE %s PASS cost=$%.6f in=%d out=%d",
            exercise.name, session_cost,
            total_tokens.get("input", 0), total_tokens.get("output", 0),
        )
        return SubsystemResult(
            name=exercise.name,
            status="pass",
            detail="All verifier checks passed.",
            duration_seconds=round(elapsed, 2),
            token_usage=total_tokens,
            cost=session_cost,
        )

    except ExerciseFailure as exc:
        elapsed = time.monotonic() - start
        return SubsystemResult(
            name=exercise.name,
            status="fail",
            detail=exc.reasoning,
            duration_seconds=round(elapsed, 2),
            error=exc.reasoning,
        )

    except Exception as exc:
        elapsed = time.monotonic() - start
        return SubsystemResult(
            name=exercise.name,
            status="error",
            detail=f"Exercise infrastructure error: {type(exc).__name__}: {exc}",
            duration_seconds=round(elapsed, 2),
            error=str(exc),
        )

    finally:
        # Cleanup temp dir
        if temp_dir is not None:
            import shutil
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


async def run_micro_exercises(
    client: "CopilotCodeClient",
    *,
    timeout: float = 600.0,
) -> ExerciseReport:
    """Run all micro-exercises sequentially and return a report."""
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()
    session_id = f"micro-exercises-{uuid.uuid4().hex[:12]}"
    start = time.monotonic()

    # Build a verifier client (uses same config — Haiku would be ideal
    # but we use whatever model is configured)
    verifier_client = client

    results: list[SubsystemResult] = []
    for exercise in MICRO_EXERCISES:
        result = await _run_single_micro_exercise(exercise, client, verifier_client)
        results.append(result)

    elapsed = time.monotonic() - start
    return ExerciseReport(
        product_name=client.config.brand.public_name,
        session_id=session_id,
        timestamp=timestamp,
        mode="micro",
        subsystems=results,
        summary=f"{sum(1 for r in results if r.status == 'pass')}/{len(results)} micro-exercises passed.",
        total_duration_seconds=round(elapsed, 2),
    )


async def run_chain_exercises(
    client: "CopilotCodeClient",
    *,
    timeout: float = 600.0,
) -> ExerciseReport:
    """Run all chain exercises sequentially and return a report."""
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()
    session_id = f"chain-exercises-{uuid.uuid4().hex[:12]}"
    start = time.monotonic()

    verifier_client = client

    results: list[SubsystemResult] = []
    for exercise in CHAIN_EXERCISES:
        result = await _run_single_micro_exercise(exercise, client, verifier_client)
        results.append(result)

    elapsed = time.monotonic() - start
    return ExerciseReport(
        product_name=client.config.brand.public_name,
        session_id=session_id,
        timestamp=timestamp,
        mode="chain",
        subsystems=results,
        summary=f"{sum(1 for r in results if r.status == 'pass')}/{len(results)} chain exercises passed.",
        total_duration_seconds=round(elapsed, 2),
    )
