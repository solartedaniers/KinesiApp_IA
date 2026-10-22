from collections.abc import AsyncIterator

from kinesiapp_ai.chat.llm import LlmClient
from kinesiapp_ai.chat.prompt_builder import AnalysisPromptInput, HistoryMessage, MovementAnalysisPromptBuilder
from kinesiapp_ai.chat.system_prompt import SYSTEM_PROMPT_VERSION


class AnalysisAssistant:
    """Facade del chat: armo el prompt del análisis y dejo que el LLM responda en streaming."""

    prompt_version = SYSTEM_PROMPT_VERSION

    def __init__(self, prompts: MovementAnalysisPromptBuilder, llm: LlmClient) -> None:
        self._prompts = prompts
        self._llm = llm

    def explain(self, analysis: AnalysisPromptInput, audience: str) -> AsyncIterator[str]:
        return self._llm.stream_reply(self._prompts.build_opening(analysis, audience))

    def answer(
        self, analysis: AnalysisPromptInput, audience: str, history: list[HistoryMessage], question: str
    ) -> AsyncIterator[str]:
        return self._llm.stream_reply(self._prompts.build(analysis, audience, history, question))
