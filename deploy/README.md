# CodeXQuant 部署包

程序提供 BTC/ETH 30 分钟事件研究信号、指数行情、企业微信提醒和 442 笔历史交易图，不提交真实订单。历史的 56.79% 是指定阶段的回测表现，当前模型已用更新的历史样本重新训练。

## Ubuntu 24.04 直接部署

1. 服务器安装 Python 3.12、python3-venv、curl。将 `codexquant-deploy.tgz`、`deploy/install.sh` 上传服务器。
2. 将桌面包里的 `runtime.env` 私下上传到 `/etc/codexquant.env`，权限设为 `600`。不要上传到代码仓库。
3. 执行 `sudo bash /路径/install.sh /路径/codexquant-deploy.tgz`。
4. 访问 `http://服务器IP/demo/trades.html`，状态接口为 `/api/live/status`，健康接口 `/health`。

安装器会先在 8088 端口检查图表和模型，再备份现有 `ecq` 服务及其程序目录，最后切换 80 端口。443 端口服务不做改动。启动失败会恢复原服务。备份保存在 `/var/backups/codexquant/时间戳`；新版本目录的 `deploy/rollback.sh` 可手动回退。

## 运行规则

- 每半小时边界后约 5 秒，读取已收盘分钟数据，产生 BTC/ETH 模型信号。
- 分数至少 0.56，方向与近 60 分钟走势相反，标准化波动达到门槛且短期波动未扩张时，进入纸面风控。
- 参考开仓时刻是信号后一分；到期是参考开仓后 30 分钟。提醒不代表实际成交。行情获取超过时限时跳过，不补推过期交易信号。
- 纸面资金初始 1,000 USDT，每笔 5 USDT、共享一笔持仓、每日亏损本金及未结算风险预算 25 USDT。信号阶段保守预留未结算风险，可能比事后资金回测少发提醒；同时信号固定 BTC 优先。
- 企业微信即时提醒标注 BTC/ETH、多/空、参考指数、北京时间和到期时刻。每小时整点播报两币指数价格，全天运行。
- 已确认发送的消息跨重启去重；失败有次数限制和重试。网络丢失回执时存在重复投递可能，状态会保留不确定标记。
- 每个半小时的全部特征（包括无交易信号）会持久保存并补记指数结算标签。每 30 天用最近 180 天已结算样本更新模型，35 天仍未更新则停止新信号。
- 行情每 15 秒更新一次，页面每 10 秒读取服务端状态。

## 运维

```bash
systemctl status ecq --no-pager
journalctl -u ecq -n 60 --no-pager
curl http://127.0.0.1/health
curl http://127.0.0.1/api/live/status
systemctl restart ecq
```

运行数据在 `/var/lib/codexquant`，模型、SQLite 和机器人配置均不通过网页发布。将 `runtime.env` 配置好后，WorkBuddy 可使用 SSH/SFTP 上传包并执行上述安装命令，无需在聊天或日志中回显密码。
