-- 第八章增量表；无外键，不修改现有会话、工单与 checkpoint。
CREATE TABLE IF NOT EXISTS tool_audit_logs (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    conversation_id VARCHAR(36) NOT NULL,
    tool_call_id VARCHAR(128) NOT NULL,
    tool_name VARCHAR(255) NOT NULL,
    source VARCHAR(255) NOT NULL,
    arguments JSON NOT NULL,
    result_summary TEXT NOT NULL,
    status VARCHAR(16) NOT NULL COMMENT '成功、失败、超时、校验拦下、权限拒绝',
    error TEXT NULL,
    retry_count INT NOT NULL DEFAULT 0,
    duration_ms INT NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX ix_tool_audit_logs_conversation_id (conversation_id),
    INDEX ix_tool_audit_logs_tool_call_id (tool_call_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
