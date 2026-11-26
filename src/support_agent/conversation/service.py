"""管理完整历史的模型投影、摘要预算和单调锚点。"""

import json
import logging

from support_agent.platform.context import ContextBudgetExceeded, estimate_tokens
from support_agent.platform.messages import context_data

from .context import ContextBudget, project_history, routing_history
from .summaries import SummaryManager

logger = logging.getLogger(__name__)


class ConversationHistory:
    def __init__(self, model, conversation, config):
        """绑定会话公开存储与摘要模型，摘要在事务外后台生成。"""
        self.repo, self.config = conversation, config
        self.summaries = SummaryManager(model, conversation, config)
        budget = ContextBudget.from_settings(config)
        logger.info(
            "context budget window=%s history=%s layer1=%s layer2=%s "
            "fixed=%s peak=%s chars_per_token=%s",
            config.context_window_tokens,
            budget.history,
            budget.layer1,
            budget.layer2,
            budget.fixed,
            budget.peak,
            config.token_chars_per_token,
        )

    async def project(self, conversation_id, request_id, history, current_msg_id):
        """生成当前会话的独立滑窗；保留原文并按预算触发追加摘要。"""
        audit = await self.repo.read_conversation(conversation_id)
        summary_text = "\n".join(s["content"] for s in audit["summaries"])
        budget = ContextBudget.from_settings(self.config)
        extra = max(
            0, estimate_tokens(summary_text, self.config) - self.config.summary_injection_tokens
        )
        available = max(0, budget.history - extra)
        if not available:
            raise ContextBudgetExceeded("上下文预算不足，请联调窗口、输入、输出及工具峰值配置。")
        view = project_history(
            history,
            self.config,
            summary_upto=audit["summary_upto_msg_id"],
            layer1_from=audit["layer1_from_msg_id"],
            current_id=current_msg_id,
            history_budget=available,
        )
        old = audit["layer1_from_msg_id"]
        if old and view.layer1_from > old and view.retired:
            logger.info("层1 降级 %s→%s cid=%s", old, view.layer1_from, conversation_id)
        await self.repo.move_layer1(conversation_id, view.layer1_from)
        if view.layer2_tokens > int(available * 0.3):
            logger.info(
                "summary trigger 层2 约 %s token > 预算 %s cid=%s from=%s upto=%s elapsed=0",
                view.layer2_tokens,
                int(available * 0.3),
                conversation_id,
                view.retired[0].additional_kwargs.get("msg_id"),
                max(
                    m.additional_kwargs.get("msg_id", m.additional_kwargs["turn_id"])
                    for m in view.retired
                ),
            )
            self.summaries.schedule(
                conversation_id,
                view.retired,
                audit["summaries"],
                audit["summary_upto_msg_id"],
            )
        logger.info(
            "history_ctx %s",
            json.dumps(
                {
                    "conversation_id": conversation_id,
                    "request_id": request_id,
                    "summaries": summary_text,
                    "window": context_data(view.window),
                    "rendered_history": context_data(routing_history(view.window)),
                    "window_count": len(view.window),
                    "tokens": estimate_tokens(view.window, self.config),
                },
                ensure_ascii=False,
            ),
        )
        return view, summary_text
