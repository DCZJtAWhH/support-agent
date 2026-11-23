-- ch07：当前会话上下文。适用于未升级的既有数据库，启动迁移也会自动补齐缺失列。
-- 不移动/删除原文，不在 conversations/messages 中储存摘要正文。
-- id <= summary_upto_msg_id：已摘要；summary_upto < id < layer1_from：层2；
-- id >= layer1_from_msg_id：层1（包含起点）。锚点按用户轮次开头移动。
-- 新工具结果存 LangGraph checkpoint，层2用量包含这些工具结果。
SET NAMES utf8mb4;

ALTER TABLE conversations
  ADD COLUMN summary_upto_msg_id INT NULL COMMENT '摘要覆盖的最后一条消息，包含终点',
  ADD COLUMN layer1_from_msg_id INT NULL COMMENT '最近原文层的第一条用户消息，包含起点';

CREATE TABLE IF NOT EXISTS conversation_summaries (
  id INT NOT NULL AUTO_INCREMENT,
  conversation_id VARCHAR(36) NOT NULL COMMENT '与 conversations.id UUID 类型一致',
  seq INT NOT NULL COMMENT '本会话第几段，从1开始，只追加',
  from_msg_id INT NOT NULL COMMENT '本段覆盖起点，包含',
  upto_msg_id INT NOT NULL COMMENT '本段覆盖终点，包含',
  content TEXT NOT NULL COMMENT '本批历史的一次性事实梗概，不回炉重写',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv_upto (conversation_id, upto_msg_id),
  CONSTRAINT fk_summary_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='当前会话分段摘要';

-- 已由应用启动迁移过的数据库无需再次执行 ALTER。旧草稿已建的 summary 列不再使用。
