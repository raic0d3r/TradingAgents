from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    opponent_argument_or_opening,
)


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = opponent_argument_or_opening(
            investment_debate_state.get("current_response", ""), "bull analyst"
        )
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        instrument_context = get_instrument_context_from_state(state)

        asset_type = state.get("asset_type", "stock")
        is_gold = bool(state.get("gold_macro_report"))

        if is_gold:
            target_label = "gold / XAUUSD"
            research_label = "Gold macro report"
            research_report = state["gold_macro_report"]
        else:
            target_label = "stock" if asset_type == "stock" else "asset"
            research_label = "Company fundamentals report"
            research_report = state.get("fundamentals_report", "")

        prompt = f"""You are a Bear Analyst making the case against investing in the {target_label}. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Identify the strongest evidence that could cause the {target_label} to decline, including financial, market, macroeconomic, geopolitical, or asset-specific risks.
- Asset-Specific Weaknesses: For stocks, consider earnings, financial health, industry trends, competitive weaknesses, and company-specific risks. For gold, focus on monetary policy, real yields, interest-rate expectations, US dollar strength, inflation, geopolitical developments, central-bank demand, and other macro drivers that could pressure XAUUSD.
- Negative Indicators: Use evidence from market data, sentiment, research, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

{instrument_context}
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{research_label}: {research_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}

Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the {target_label}.
""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node

