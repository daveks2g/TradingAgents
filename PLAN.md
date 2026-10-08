# Plan: C_Agents NinjaTrader 8 Strategy + Python Communication

## Context
The Python multi-agent ML trading system needs to receive 5-min bar data from NinjaTrader and send back signals (regime, signal, valuation, stop-loss, target) to display in the chart trader panel. The strategy writes bar data to a CSV for Python to consume, and reads back a separate CSV with agent decisions.

## Communication Design: Dual CSV File Exchange

**NT8 → Python:** `data_files/bars.csv` (strategy appends each new bar)
**Python → NT8:** `data_files/agent_signals.csv` (Python overwrites with latest signals)

The signal CSV will have a single row (plus header) that gets overwritten on each bar:
```
bar_index,regime,regime_conf,signal,signal_conf,valuation_score,valuation_conf,stop_distance,stop_price_long,stop_price_short,target_price_long,target_price_short
```

NT8 reads this file on each bar update using a simple `File.ReadAllLines` with try/catch for file locks. This is simpler and more reliable than sockets/pipes for this use case.

## File to Create
`C:\Users\davek\OneDrive\Documents\NinjaTrader 8\bin\Custom\Strategies\A_DK\C_Agents.cs`

## Strategy Structure

### OnStateChange
- **SetDefaults:** Name, Calculate=OnBarClose, strategy params (CsvOutputPath, CsvSignalPath, LightSize, FontSize)
- **Configure:** Nothing special
- **DataLoaded:** Create chart trader WPF panel via Dispatcher
- **Terminated:** Cleanup WPF panel via Dispatcher

### OnBarUpdate
1. Early return if `CurrentBar < BarsRequiredToTrade` or `BarsInProgress != 0`
2. Build CSV line: `datetime,open,high,low,close,volume`
3. Append to `bars.csv` using StreamWriter (append=true), write header if file doesn't exist
4. Read `agent_signals.csv` with try/catch (file may be locked by Python)
5. Parse signal values into fields
6. Update WPF lights on Dispatcher thread

### Chart Trader Panel (WPF)
Based on patterns from `A_ChecklistMine.cs` and `C_Integration.cs`:

**Layout:** Single column of lights + labels, no buttons (read-only display)

```
[Light] Regime: Trending Up
[Light] Signal: Buy
[Light] Valuation: Oversold
        SL: 4520.25
        Target: 4535.50
```

**Grid structure:** 3 columns
- Col 0: Ellipse (light, size LightSize)
- Col 1: 5px spacer
- Col 2: Label text (star width)

**Rows:**
- Row 0: Regime light + label (color: green=trending_up, red=trending_down, yellow=ranging, gray=chop)
- Row 1: Signal light + label (green=buy, red=sell, gray=hold)
- Row 2: Valuation light + label (green=oversold, red=overbought, yellow=neutral)
- Row 3: "SL:" label + value TextBlock
- Row 4: "Target:" label + value TextBlock

**Color mapping:**
- Regime: Lime=trending_up, Red=trending_down, Yellow=ranging, DimGray=chop
- Signal: Lime=buy, Red=sell, DimGray=hold
- Valuation: Lime=oversold(<-0.3), Red=overbought(>0.3), Yellow=neutral

**Methods:**
- `CreateAgentsPanel()` -- builds WPF grid, adds to chart trader (same pattern as C_Integration.CreateChartTraderLabels)
- `RemoveAgentsPanel()` -- removes from chart trader grid, nulls references (same pattern as C_Integration.RemoveChartTraderLabels)
- `UpdatePanelDisplay()` -- sets ellipse fills and label text from parsed signal data

### Properties
- `CsvOutputPath` (string) -- path to bars CSV output, default to TradingAgents/data_files/bars.csv
- `CsvSignalPath` (string) -- path to agent signals CSV input, default to TradingAgents/data_files/agent_signals.csv
- `LightSize` (int, 10-40, default 18)
- `FontSize` (int, 8-20, default 14)

## Python-side Changes
Update `orchestrator.py` to write `agent_signals.csv` after each bar processed, with the latest agent outputs. Simple `csv.writer` overwrite (write header + 1 data row). Include target price calculation (close + stop_distance for profit target, using 1.5:1 reward-risk).

## Files Modified
1. **NEW:** `C:\Users\davek\OneDrive\Documents\NinjaTrader 8\bin\Custom\Strategies\A_DK\C_Agents.cs`
2. **EDIT:** `g:\source\repos\MyPythonProjects\TradingAgents\orchestrator.py` -- add signal CSV writer
3. **EDIT:** `g:\source\repos\MyPythonProjects\TradingAgents\config.py` -- add SIGNAL_CSV_PATH

## Verification
1. Add strategy to NinjaTrader chart -- should compile and show panel in chart trader area
2. Run with historical data -- `bars.csv` should populate with OHLCV rows
3. Run Python `main.py` -- should read bars, compute signals, write `agent_signals.csv`
4. NT8 panel should update lights and SL/target values from the signal file
