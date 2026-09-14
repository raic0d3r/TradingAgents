from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)


def create_gold_macro_analyst(llm):
    def gold_macro_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_macro_indicators,
            get_global_news,
            get_news,
            get_prediction_markets,
        ]

        system_message = (
            "You are a Gold Macro Analyst responsible for analyzing the "
            "macroeconomic and geopolitical forces that drive gold and XAUUSD. "
            "Your analysis must focus on the specific instrument identified "
            "in the instrument context and the specified analysis date. "

            "Analyze the major macro drivers of gold, including Federal Reserve "
            "monetary policy, interest-rate expectations, US Treasury yields, "
            "real interest rates, inflation, employment and economic growth, "
            "US dollar strength, global risk sentiment, geopolitical risks, "
            "central-bank policy and gold demand, and other major factors that "
            "can materially affect gold prices. "

            "Use the available macroeconomic, global-news, and prediction-market "
            "tools to gather evidence. Distinguish between confirmed data, "
            "market expectations, and your own interpretation. Pay particular "
            "attention to developments that could change the direction of "
            "XAUUSD over the relevant trading horizon. "

            "Explain whether each major factor is bullish, bearish, or neutral "
            "for gold and why. Identify important upcoming catalysts, risks, "
            "and potential changes in the macro regime. "

            "Do not analyze company financial statements, revenue, earnings, "
            "balance sheets, cash flow, or other corporate fundamentals. "
            "Gold is a macro-driven asset rather than a company. "

            "Provide specific, actionable insights with supporting evidence "
            "to help downstream traders make informed decisions. "

            "At the end of the report, append a Markdown table summarizing "
            "the key macro drivers, their current direction, supporting "
            "evidence, and implications for XAUUSD."

            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other "
                    "assistants. Use the provided tools to progress towards "
                    "answering the question. If you are unable to fully answer, "
                    "that's OK; another assistant with different tools will "
                    "help where you left off. "

                    "You have access to the following tools: {tool_names}. "
                    "Today's date is {current_date}; treat it as 'now' for all "
                    "analysis and tool-call date ranges. "
                    "{instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(
            tool_names=", ".join([tool.name for tool in tools])
        )
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "gold_macro_report": report,
        }

    return gold_macro_analyst_node