#!/usr/bin/env python3
"""
135均线交易系统 — 分仓建仓 + 移动止损回测 v14
==========================================
版本: v14 — 修复trade.dt获取问题
日期: 2026-08-14

修复:
  1. 用order.executed.dt获取成交日期（在notify_order中记录）
  2. 持仓天数改用实际日期差
  3. 移除调试打印
"""

import argparse
import datetime
import os
import sys
import tempfile

import backtrader as bt
import baostock as bs
import pandas as pd


# ============================================================
# 数据获取
# ============================================================

def get_hs300_stocks():
    try:
        bs.login()
        rs = bs.query_hs300_stocks()
        codes = []
        while rs is not None and rs.error_code == '0' and rs.next():
            codes.append(rs.get_row_data()[1] if len(rs.get_row_data()) > 1 else rs.get_row_data()[0])
        bs.logout()
        return codes
    except Exception as e:
        print(f"获取沪深300失败: {e}")
        return []


def get_stock_data(code, start_date, end_date):
    try:
        bs.login()
        rs = bs.query_history_k_data_plus(
            code, "date,open,high,low,close,volume",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"
        )
        data = []
        while rs and rs.error_code == '0' and rs.next():
            data.append(rs.get_row_data())
        bs.logout()
        df = pd.DataFrame(data, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna(subset=['close', 'volume']).sort_index()
    except Exception as e:
        return None


def df_to_csv(df, path):
    df = df.copy()
    df.index.name = 'date'
    df = df.reset_index()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df.to_csv(path, index=False, float_format='%.4f')


# ============================================================
# 策略定义
# ============================================================

class Strategy135Position(bt.Strategy):
    """
    分仓建仓 + 移动止损策略
    
    分仓逻辑(按最大可买股数分10份):
      试仓: max_shares / 10 (最低100股)
      确认: + 3 × max_shares / 10 (持仓>=3天)
      满仓: + 5 × max_shares / 10 (MA13>MA34>MA55)
    """

    params = (
        ('ma13', 13), ('ma34', 34), ('ma55', 55),
        ('stop_loss_pct', 0.07), ('commission', 0.001),
        ('signal_cooldown', 5),
        ('stage2_days', 3),
    )

    def __init__(self):
        self.ma13 = bt.indicators.SMA(self.data.close, period=self.p.ma13)
        self.ma34 = bt.indicators.SMA(self.data.close, period=self.p.ma34)
        self.ma55 = bt.indicators.SMA(self.data.close, period=self.p.ma55)

        self.signals = []
        self.trade_log = []
        
        self.position_stage = 0  # 0=空, 1=试仓, 2=5份, 3=10份
        self.high_since_entry = 0
        self.hold_days = 0
        self.last_buy_bar = -999
        
        self.max_shares = 0
        self.initial_cash = self.broker.getcash()
        self.hold_start_bar = -999
        self.entry_date = None  # 只在首次买入时设置
        self.position_opened = False  # 标记是否已开仓

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.high_since_entry = max(self.high_since_entry, order.executed.price)
                # FIX: 只在首次开仓时记录entry_date
                if not self.position_opened:
                    self.entry_date = bt.num2date(self.data.datetime[0]).strftime('%Y-%m-%d')
                    self.position_opened = True

    def notify_trade(self, trade):
        if trade.isclosed:
            # FIX: trade.size=0(平仓后净头寸), trade.price=加权平均价
            # 真实盈亏用trade.pnlcomm(含佣金)
            pnl = trade.pnlcomm if trade.pnlcomm else 0
            
            # FIX: 使用self.entry_date而不是trade.dt（backtrader 1.9.78中trade.dt为None）
            entry_dt = self.entry_date if self.entry_date else 'N/A'
            exit_dt = self.data.datetime[0]
            
            self.trade_log.append({
                'entry_date': entry_dt,
                'exit_date': bt.num2date(exit_dt).strftime('%Y-%m-%d'),
                'entry_price': trade.price,
                'exit_stage': self.position_stage,
                'pnl': round(pnl, 2),
                'return_pct': round(pnl / self.initial_cash * 100, 2),
                'hold_days': self.hold_days,
            })

    def next(self):
        # 需要足够的历史数据计算MA55
        if len(self.data) < self.p.ma55:
            return

        price = self.data.close[0]
        bar_idx = len(self.data)
        
        # 更新持仓状态
        if self.position:
            self.high_since_entry = max(self.high_since_entry, price)
            if self.hold_start_bar >= 0:
                self.hold_days = bar_idx - self.hold_start_bar
        else:
            self.hold_days = 0

        # 计算最大可买股数
        current_cash = self.broker.getcash()
        new_max = int(current_cash * 0.95 / price / 100) * 100
        if new_max > 0:
            self.max_shares = new_max

        # ========================================
        # 持仓中: 止损/卖出
        # ========================================
        if self.position:
            # 移动止损7%
            stop = self.high_since_entry * (1 - self.p.stop_loss_pct)
            if price <= stop:
                self.close()
                self.signals.append({
                    'date': bt.num2date(self.data.datetime[0]).strftime('%Y-%m-%d'),
                    'type': 'sell',
                    'signal': f'移动止损(最高={self.high_since_entry:.2f})',
                    'price': price,
                    'stage': self.position_stage
                })
                return

            # 135卖出信号
            sell = self._detect_sells()
            if sell:
                self.close()
                self.signals.append({
                    'date': bt.num2date(self.data.datetime[0]).strftime('%Y-%m-%d'),
                    'type': 'sell',
                    'signal': sell,
                    'price': price,
                    'stage': self.position_stage
                })
                return

        # ========================================
        # 空仓: 买入(试仓)
        # ========================================
        if not self.position and self.position_stage == 0:
            if bar_idx - self.last_buy_bar < self.p.signal_cooldown:
                return
            if self.max_shares < 100:
                return

            buy = self._detect_buys()
            if buy:
                stage1 = max(100, self.max_shares // 10)
                stage1 = (stage1 // 100) * 100
                if stage1 >= 100:
                    self.buy(size=stage1)
                    self.position_stage = 1
                    self.last_buy_bar = bar_idx
                    self.hold_start_bar = bar_idx
                    self.signals.append({
                        'date': bt.num2date(self.data.datetime[0]).strftime('%Y-%m-%d'),
                        'type': 'buy',
                        'signal': buy,
                        'price': price,
                        'stage': 1,
                        'stage_name': f'试仓({stage1}股)'
                    })

        # ========================================
        # 加仓: 试仓→确认(加3份)
        # ========================================
        if self.position and self.position_stage == 1 and self.hold_days >= self.p.stage2_days:
            add_size = max(100, self.max_shares // 10 * 3)
            add_size = (add_size // 100) * 100
            if add_size >= 100:
                self.buy(size=add_size)
                self.position_stage = 2
                self.signals.append({
                    'date': bt.num2date(self.data.datetime[0]).strftime('%Y-%m-%d'),
                    'type': 'buy',
                    'signal': '确认加仓',
                    'price': price,
                    'stage': 2,
                    'stage_name': f'确认(加{add_size}股)'
                })

        # ========================================
        # 加仓: 确认→满仓(加5份)
        # ========================================
        if self.position and self.position_stage == 2:
            if self.ma13[0] > self.ma34[0] > self.ma55[0]:
                add_size = max(100, self.max_shares // 10 * 5)
                add_size = (add_size // 100) * 100
                if add_size >= 100:
                    self.buy(size=add_size)
                    self.position_stage = 3
                    self.signals.append({
                        'date': bt.num2date(self.data.datetime[0]).strftime('%Y-%m-%d'),
                        'type': 'buy',
                        'signal': '趋势满仓',
                        'price': price,
                        'stage': 3,
                        'stage_name': f'满仓(加{add_size}股)'
                    })

    # ============================================================
    # 买入信号
    # ============================================================
    def _detect_buys(self):
        cands = []
        if (self.ma13[-1] < self.ma55[-1] and self.ma13[0] > self.ma55[0] and
            self.data.close[0] > self.data.open[0]):
            cands.append(("MA13金叉MA55", "高"))
        if self.ma13[0] > self.ma34[0] > self.ma55[0] and self.data.close[0] > self.data.open[0]:
            cands.append(("均线多头排列", "中"))
        if (self.ma13[-1] <= self.ma13[-2] and self.ma13[0] > self.ma13[-1] and
            self.data.close[0] > self.ma13[0] and self.data.close[0] > self.data.open[0]):
            cands.append(("红杏出墙", "高"))
        if (self.data.close[-1] < self.ma55[-1] and self.data.close[0] > self.ma55[0] and
            self.data.close[0] > self.data.open[0]):
            cands.append(("放量突破MA55", "中"))
        
        if not cands:
            return None
        order_map = {"高": 0, "中": 1, "低": 2}
        cands.sort(key=lambda x: order_map.get(x[1], 3))
        return cands[0][0]

    # ============================================================
    # 卖出信号
    # ============================================================
    def _detect_sells(self):
        cands = []
        if (self.ma13[-1] > self.ma55[-1] and self.ma13[0] < self.ma55[0] and
            self.data.close[0] < self.ma13[0]):
            cands.append("MA13死叉MA55")
        if (self.ma13[0] < self.ma34[0] < self.ma55[0] and
            self.data.close[0] < self.data.close[-1]):
            cands.append("空头排列下跌")
        if (self.data.close[0] < self.ma13[0] and self.data.close[0] < self.ma34[0] and
            self.data.close[0] < self.ma55[0] and self.data.close[0] < self.data.open[0]):
            cands.append("一阴破三线")
        return cands[0] if cands else None


# ============================================================
# 回测执行
# ============================================================

def run_single_backtest(code, initial_cash=1000000,
                        start_date="2023-01-01", end_date="2026-08-13",
                        stop_loss=0.07):
    """单只股票回测"""
    df = get_stock_data(code, start_date, end_date)
    if df is None or len(df) < 60:
        return None, f"数据不足"
    
    # 获取股票名称
    name = ""
    try:
        bs.login()
        rs = bs.query_stock_basic(code)
        while rs.next():
            r = rs.get_row_data()
            name = r[2] if len(r) > 2 else ""
        bs.logout()
    except:
        pass

    # 导出CSV供backtrader使用
    tmpfile = None
    try:
        tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        tmpfile.close()
        df_to_csv(df, tmpfile.name)

        cerebro = bt.Cerebro()
        # 使用CSV文件加载数据
        data_feed = bt.feeds.GenericCSVData(
            dataname=tmpfile.name,
            dtformat='%Y-%m-%d',
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,
            headers=True
        )
        cerebro.adddata(data_feed)
        
        cerebro.addstrategy(Strategy135Position, stop_loss_pct=stop_loss)
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.03, annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

        results = cerebro.run()
        strat = results[0]

        signals = strat.signals
        trade_log = strat.trade_log
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash * 100

        total_trades = len(trade_log)
        returns_list = []
        total_hold_days = 0
        best_ret = -999
        worst_ret = 999
        stage_3_count = 0
        stage_sum = 0

        for t in trade_log:
            ret = t.get('return_pct', 0)
            if isinstance(ret, (int, float)):
                returns_list.append(ret)
                if ret > best_ret: best_ret = ret
                if ret < worst_ret: worst_ret = ret
            entry_d = t.get('entry_date', '')
            exit_d = t.get('exit_date', '')
            if isinstance(entry_d, str) and isinstance(exit_d, str) and entry_d != 'N/A':
                try:
                    d1 = datetime.datetime.strptime(entry_d, '%Y-%m-%d')
                    d2 = datetime.datetime.strptime(exit_d, '%Y-%m-%d')
                    total_hold_days += abs((d2 - d1).days)
                except:
                    pass
            stage = t.get('exit_stage', 1)
            stage_sum += stage
            if stage >= 3: stage_3_count += 1

        win_trades = [r for r in returns_list if r > 0]
        lose_trades = [r for r in returns_list if r <= 0]
        win_rate = len(win_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_hold = total_hold_days / total_trades if total_trades > 0 else 0

        sharpe = 0
        sa = strat.analyzers.sharpe.get_analysis()
        if isinstance(sa, dict) and 'sharperatio' in sa:
            v = sa['sharperatio']
            if isinstance(v, (int, float)): sharpe = v

        max_dd = 0
        da = strat.analyzers.drawdown.get_analysis()
        if isinstance(da, dict):
            m = da.get('max', {})
            max_dd = m.get('drawdown', 0) if isinstance(m, dict) else m

        annual_ret = 0
        ra = strat.analyzers.returns.get_analysis()
        if isinstance(ra, dict) and 'rtot' in ra:
            annual_ret = ra['rtot'] * 100
        if annual_ret == 0 and total_trades == 0:
            try:
                years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
                if years > 0:
                    annual_ret = ((final_value / initial_cash) ** (1 / years) - 1) * 100
            except:
                pass

        buy_sigs = [s for s in signals if s.get('type') == 'buy']
        sell_sigs = [s for s in signals if s.get('type') == 'sell']
        sig_dist = {}
        for s in signals:
            name_s = s.get('signal', 'unknown')
            sig_dist[name_s] = sig_dist.get(name_s, 0) + 1

        return {
            'code': code,
            'name': name,
            'final_value': round(final_value, 2),
            'total_return_pct': round(total_return, 2),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown_pct': round(max_dd, 2),
            'annual_return_pct': round(annual_ret, 2),
            'total_trades': total_trades,
            'win_count': len(win_trades),
            'lose_count': len(lose_trades),
            'win_rate_pct': round(win_rate, 1),
            'buy_signals': len(buy_sigs),
            'sell_signals': len(sell_sigs),
            'avg_hold_days': round(avg_hold, 1),
            'best_trade_pct': round(best_ret if returns_list else 0, 2),
            'worst_trade_pct': round(worst_ret if returns_list else 0, 2),
            'stage_3_reached': stage_3_count,
            'avg_stage': round(stage_sum / total_trades, 2) if total_trades > 0 else 0,
            'signal_distribution': str(sig_dist),
            'data_start': str(df.index[0].date()),
            'data_end': str(df.index[-1].date()),
            'data_bars': len(df),
            'error': '',
            'signals': signals,
            'trade_log': trade_log,
        }, None

    except Exception as e:
        return None, str(e)
    finally:
        if tmpfile and os.path.exists(tmpfile.name):
            try:
                os.unlink(tmpfile.name)
            except:
                pass


def print_single_result(code, result):
    """打印单只股票回测结果"""
    print(f"\n{'='*60}")
    print(f"  {code} {result['name']}")
    print(f"{'='*60}")
    print(f"  最终价值:       {result['final_value']:,.2f}")
    print(f"  总收益率:       {result['total_return_pct']:+.2f}%")
    print(f"  夏普比率:       {result['sharpe_ratio']:.3f}")
    print(f"  最大回撤:       {result['max_drawdown_pct']:.2f}%")
    print(f"  年化收益:       {result['annual_return_pct']:.2f}%")
    print(f"  总交易:         {result['total_trades']}笔")
    print(f"  胜率:           {result['win_rate_pct']:.1f}%")
    print(f"  买入信号:       {result['buy_signals']}")
    print(f"  卖出信号:       {result['sell_signals']}")
    print(f"  平均持仓:       {result['avg_hold_days']:.1f}天")
    print(f"  最佳交易:       {result['best_trade_pct']:+.2f}%")
    print(f"  最差交易:       {result['worst_trade_pct']:+.2f}%")
    print(f"  满仓次数:       {result['stage_3_reached']}")
    print(f"  平均仓位:       {result['avg_stage']:.2f}")
    print(f"  信号分布:       {result['signal_distribution']}")
    
    # 打印详细信号
    print(f"\n  --- 详细信号 ---")
    for sig in result['signals']:
        print(f"    {sig['date']} | {sig['type']:4s} | {sig['signal']} | stage={sig['stage']}")
    
    # 打印交易记录
    print(f"\n  --- 交易记录 ---")
    for trade in result['trade_log']:
        print(f"    {trade['entry_date']} → {trade['exit_date']} | "
              f"价:{trade['entry_price']:.2f} | 阶段:{trade['exit_stage']} | "
              f"盈亏:{trade['pnl']:+,.2f}({trade['return_pct']:+.2f}%) | "
              f"持仓:{trade['hold_days']}天")
    
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, default=None)
    parser.add_argument('--top', type=int, default=300)
    parser.add_argument('--start', type=str, default='2023-01-01')
    parser.add_argument('--end', type=str, default='2026-08-13')
    parser.add_argument('--output', type=str, default='position_backtest_v14.csv')
    parser.add_argument('--results', type=int, default=50)
    parser.add_argument('--capital', type=float, default=1000000)
    parser.add_argument('--stop-loss', type=float, default=0.07)
    args = parser.parse_args()

    if args.code:
        result, error = run_single_backtest(
            args.code, initial_cash=args.capital,
            start_date=args.start, end_date=args.end,
            stop_loss=args.stop_loss
        )
        if error:
            print(f"错误: {error}")
            return
        print_single_result(args.code, result)
        return

    # 批量回测
    codes = get_hs300_stocks()[:args.top]
    if not codes:
        print("无法获取股票池")
        return

    print(f"批量回测 {len(codes)} 只股票...")
    results = []
    for i, code in enumerate(codes):
        result, error = run_single_backtest(
            code, initial_cash=args.capital,
            start_date=args.start, end_date=args.end,
            stop_loss=args.stop_loss
        )
        if result:
            results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(codes)}")

    df = pd.DataFrame(results)
    if df.empty:
        print("没有回测结果")
        return
    
    df = df.sort_values('total_return_pct', ascending=False)
    top_df = df.head(args.results)

    print(f"\n回测完成! Top {len(top_df)}:")
    cols = ['code', 'name', 'total_return_pct', 'total_trades', 'win_rate_pct',
            'sharpe_ratio', 'max_drawdown_pct', 'buy_signals', 'stage_3_reached']
    print(top_df[cols].to_string(index=False))

    out = args.output
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {out}")


if __name__ == '__main__':
    main()
