
from tradingagents.graph.analyst_execution import build_analyst_execution_plan

print('Gold:',
      [s.key for s in build_analyst_execution_plan(
          ('market', 'social', 'news', 'gold_macro')
      ).specs])

print('Stock:',
      [s.key for s in build_analyst_execution_plan(
          ('market', 'social', 'news', 'fundamentals')
      ).specs])
