from __future__ import annotations

from .agents import build_default_custom_agents, persist_agent_output, COORDINATOR_PROTOCOL, WORKTREE_NOTICE
from .brand import BrandSpec, DEFAULT_BRAND
from .client import CopilotCodeClient, CopilotCodeSession
from .config import CopilotCodeConfig, DEFAULT_AGENT_NAMES, DEFAULT_SKILL_NAMES
from .memory import MemoryStore
from .permissions import PermissionPolicy
from .prompt_compiler import (
    CYBER_RISK_INSTRUCTION,
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    PromptAssembler,
    PromptPriority,
    PromptSection,
    build_assembler,
    build_system_message,
    gather_git_context,
    materialize_workspace_instructions,
    render_claude_md_template,
    render_copilot_instructions_template,
)
from .exercise import ExerciseReport, SubsystemResult, SUBSYSTEM_CHECKLIST, build_exercise_prompt, parse_exercise_report, run_exercise
from .extraction import ExtractionMode, SESSION_MEMORY_SECTIONS, build_enforce_extraction_prompt, build_extraction_prompt, build_session_end_extraction_prompt, build_session_memory_update_prompt, should_extract
from .session_memory import SessionMemoryController, SessionMemoryState
from .session_state import SessionState, SessionStatus, RequiresActionDetails
from .diff import DiffResult, generate_diff, generate_file_diff, summarize_changes, apply_patch
from .events import Event, EventBus, EventType
from .retry import RetryPolicy, RetryState, build_retry_response
from .subagent import ChildSession, EnforcedChildSession, MaxTurnsExceeded, SubagentContext, SubagentSpec, build_subagent_context
from .mcp import build_mcp_prompt_section, build_mcp_delta, validate_mcp_server_config, MCPLifecycleManager, MCPServerStatus, MCPServerState
from .tokenizer import count_tokens, estimate_tokens, count_message_tokens, has_tiktoken
from .compaction import CompactionMode, CompactionResult, build_compaction_prompt, build_handoff_context, parse_compaction_response
from .instructions import InstructionBundle, load_workspace_instructions
from .reports import CheckResult, PreflightReport, SmokeTestReport
from .suggestions import build_prompt_suggestions, format_suggestions_prompt
from .skill_assets import SkillTracker, build_skill_catalog, parse_skill_frontmatter
from .skill_tool import build_skill_tool, build_skill_tool_prompt
from .verifier import (
    MAX_VERIFICATION_ATTEMPTS,
    MAX_VERIFIER_MALFUNCTIONS,
    VERIFIER_SYSTEM_PROMPT,
    VerificationExhaustedError,
    VerificationResult,
    build_verifier_prompt,
    compare_output_hashes,
    extract_failed_checks,
    format_fail_feedback,
    parse_verdict,
    run_verification,
    snapshot_output_hashes,
    write_failure_trace,
)
from .tasks import TaskRecord, TaskStatus, TaskStore
from .task_tools import build_task_tools
from .token_budget import TokenBudget, parse_token_budget, strip_budget_directive, format_budget_status
from .model_cost import UsageCost, calculate_cost, get_knowledge_cutoff, MODEL_PRICING

__all__ = [
    "BrandSpec",
    "CheckResult",
    "CopilotCodeClient",
    "CopilotCodeConfig",
    "CopilotCodeSession",
    "DEFAULT_AGENT_NAMES",
    "DEFAULT_BRAND",
    "DEFAULT_SKILL_NAMES",
    "ExerciseReport",
    "SubsystemResult",
    "SUBSYSTEM_CHECKLIST",
    "build_exercise_prompt",
    "parse_exercise_report",
    "run_exercise",
    "InstructionBundle",
    "MemoryStore",
    "PermissionPolicy",
    "PromptAssembler",
    "PromptPriority",
    "PromptSection",
    "PreflightReport",
    "SmokeTestReport",
    "TaskRecord",
    "TaskStatus",
    "TaskStore",
    "CompactionMode",
    "CompactionResult",
    "build_assembler",
    "build_compaction_prompt",
    "COORDINATOR_PROTOCOL",
    "WORKTREE_NOTICE",
    "build_default_custom_agents",
    "persist_agent_output",
    "build_extraction_prompt",
    "build_session_end_extraction_prompt",
    "build_handoff_context",
    "parse_compaction_response",
    "build_mcp_prompt_section",
    "build_mcp_delta",
    "validate_mcp_server_config",
    "SkillTracker",
    "build_skill_catalog",
    "build_skill_tool",
    "build_skill_tool_prompt",
    "build_task_tools",
    "build_prompt_suggestions",
    "format_suggestions_prompt",
    "build_system_message",
    "load_workspace_instructions",
    "materialize_workspace_instructions",
    "parse_skill_frontmatter",
    "render_claude_md_template",
    "render_copilot_instructions_template",
    "SESSION_MEMORY_SECTIONS",
    "SessionMemoryController",
    "SessionMemoryState",
    "SessionState",
    "SessionStatus",
    "RequiresActionDetails",
    "build_session_memory_update_prompt",
    "should_extract",
    "CYBER_RISK_INSTRUCTION",
    "SYSTEM_PROMPT_DYNAMIC_BOUNDARY",
    "gather_git_context",
    "TokenBudget",
    "parse_token_budget",
    "strip_budget_directive",
    "format_budget_status",
    "MODEL_PRICING",
    "UsageCost",
    "calculate_cost",
    "get_knowledge_cutoff",
    "DiffResult",
    "generate_diff",
    "generate_file_diff",
    "summarize_changes",
    "apply_patch",
    "MCPLifecycleManager",
    "MCPServerStatus",
    "MCPServerState",
    "count_tokens",
    "estimate_tokens",
    "count_message_tokens",
    "has_tiktoken",
    "Event",
    "EventBus",
    "EventType",
    "RetryPolicy",
    "RetryState",
    "build_retry_response",
    "ChildSession",
    "EnforcedChildSession",
    "MaxTurnsExceeded",
    "SubagentContext",
    "SubagentSpec",
    "build_subagent_context",
    "MAX_VERIFICATION_ATTEMPTS",
    "MAX_VERIFIER_MALFUNCTIONS",
    "VERIFIER_SYSTEM_PROMPT",
    "VerificationExhaustedError",
    "VerificationResult",
    "build_verifier_prompt",
    "compare_output_hashes",
    "extract_failed_checks",
    "format_fail_feedback",
    "parse_verdict",
    "run_verification",
    "snapshot_output_hashes",
    "write_failure_trace",
]

__version__ = "0.2.0"
