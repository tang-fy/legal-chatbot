# 路由评测

- 题数: 50
- 准确率: 96.0%
- Agent 类召回: 86.7%
- Agent 类精确率: 100.0%
- Fast 误判为 Agent 的比例: 0.0%
- 明细: `holdout_route_details.csv`

## 误判样本

| id | 期望 | 实际 | 问题 |
|---|---|---|---|
| h22 | agent | fast | 王五盗窃案是哪一年判决的？ |
| h25 | agent | fast | 哪个案件被判了有期徒刑一年？ |
