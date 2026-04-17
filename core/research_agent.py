import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI 
from .interview_builder import Analyst, InterviewBuilder

import operator
from typing import List, Annotated
from typing_extensions import TypedDict

from langgraph.constants import Send
from .prompts import report_writer_instructions, intro_conclusion_instructions
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from .utils import sanitize_messages

def get_llm():
    # Use os.getenv for better handling on Streamlit Cloud
    model = os.getenv("MODEL", "gpt-4o") # Default to a safe model if not specified
    api_key = os.getenv("NOVITA_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE") or "https://api.openai.com/v1"

    if not api_key:
        import streamlit as st
        st.error("API Key missing! Please set NOVITA_API_KEY or OPENAI_API_KEY in your secrets.")
        st.stop()

    llm = ChatOpenAI(
        model=model, 
        temperature=0.7,
        openai_api_key=api_key,
        api_base=api_base,
        max_retries=5,
        timeout=60,
    )

    return llm 

class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst] 
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str

class ResearchAgent:
    def __init__(self, templatePrompt):
        self.llm = get_llm()
        self.templatePrompt = templatePrompt
        self.report_writer_instructions = report_writer_instructions
        self.intro_conclusion_instructions = intro_conclusion_instructions
        self.interview_builder = InterviewBuilder(self.llm)


    def initiate_all_interviews(self, state: ResearchGraphState):
        """ This is the "map" step where we run each interview sub-graph using Send API """

        human_analyst_feedback=state.get('human_analyst_feedback')
        if human_analyst_feedback:

            return "create_analysts"

        else:
            topic = state["topic"]
            return [Send("conduct_interview", {"analyst": analyst,
                                            "messages": [HumanMessage(
                                                content=f"So you said you were writing an article on {topic}?"
                                            )
                                                        ]}) for analyst in state["analysts"]]


    def write_report(self, state: ResearchGraphState):

        sections = state["sections"]
        topic = state["topic"]

        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        system_message = self.report_writer_instructions.format(topic=topic, context=formatted_str_sections, template=self.templatePrompt)
        full_messages = [SystemMessage(content=system_message)] + [HumanMessage(content=f"Write a report based upon these memos.")]
        sanitized = sanitize_messages(full_messages)
        report = self.llm.invoke(sanitized)
        return {"content": report.content}
    

    def write_introduction(self, state: ResearchGraphState):

        sections = state["sections"]
        topic = state["topic"]

        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

        instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
        full_messages = [SystemMessage(content=instructions)] + [HumanMessage(content=f"Write the report introduction")]
        sanitized = sanitize_messages(full_messages)
        intro = self.llm.invoke(sanitized)
        return {"introduction": intro.content}


    def write_conclusion(self, state: ResearchGraphState):

        sections = state["sections"]
        topic = state["topic"]

        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

        instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
        full_messages = [SystemMessage(content=instructions)] + [HumanMessage(content=f"Write the report conclusion")]
        sanitized = sanitize_messages(full_messages)
        conclusion = self.llm.invoke(sanitized)
        return {"conclusion": conclusion.content}


    def finalize_report(self, state: ResearchGraphState):
        """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """

        content = state["content"]
        if content.startswith("## Insights"):
            content = content
            
        if "## Sources" in content:
            try:
                content, sources = content.split("\n## Sources\n")
            except:
                sources = None
        else:
            sources = None

        final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
        if sources is not None:
            final_report += "\n\n## Sources\n" + sources
        return {"final_report": final_report}
    

    def build(self):
        builder = StateGraph(ResearchGraphState)
        
        # Define a standard retry policy for transient API errors
        retry_policy = RetryPolicy(max_attempts=3, backoff_factor=2.0)
        
        builder.add_node("create_analysts", self.interview_builder.create_analysts, retry=retry_policy)
        builder.add_node("human_feedback",  self.interview_builder.human_feedback)
        builder.add_node("conduct_interview", self.interview_builder.build().compile(), retry=retry_policy)
        builder.add_node("write_report",self.write_report, retry=retry_policy)
        builder.add_node("write_introduction",self.write_introduction, retry=retry_policy)
        builder.add_node("write_conclusion",self.write_conclusion, retry=retry_policy)
        builder.add_node("finalize_report",self.finalize_report, retry=retry_policy)

        builder.add_edge(START, "create_analysts")
        builder.add_edge("create_analysts", "human_feedback")
        builder.add_conditional_edges("human_feedback", self.initiate_all_interviews, ["create_analysts", "conduct_interview"])
        builder.add_edge("conduct_interview", "write_report")
        builder.add_edge("conduct_interview", "write_introduction")
        builder.add_edge("conduct_interview", "write_conclusion")
        builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
        builder.add_edge("finalize_report", END)

        memory = MemorySaver()
        graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
        return graph
