
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import START, END, StateGraph

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from .prompts import analyst_instructions, question_instructions, search_instructions, answer_instructions, section_writer_instructions

import operator
from typing import  Annotated
from langgraph.graph import MessagesState

from langchain_core.messages import get_buffer_string

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
import asyncio
import json
import random
import time
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.types import RetryPolicy
from .utils import sanitize_messages

def apply_jitter(min_s=0.5, max_s=1.5):
    """ Small random delay to avoid provider-side traffic spikes during parallel nodes """
    time.sleep(random.uniform(min_s, max_s))


class Analyst(BaseModel):
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Role: {self.role}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )


class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]


class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class InterviewBuilder:
    
    def __init__(self, llm):
        self.llm = llm
        self.analyst_instructions = analyst_instructions
        self.question_instructions = question_instructions
        self.answer_instructions = answer_instructions
        self.section_writer_instructions = section_writer_instructions
        self.tavily_search = TavilySearchResults(max_results=3)
        self.search_instructions = search_instructions

    def create_analysts(self, state: GenerateAnalystsState):

        """ Create analysts """
        topic=state['topic']
        max_analysts=state['max_analysts']
        human_analyst_feedback=state.get('human_analyst_feedback', '')
        
        parser = PydanticOutputParser(pydantic_object=Perspectives)
        
        system_message = analyst_instructions.format(topic=topic,
                                                    human_analyst_feedback=human_analyst_feedback,
                                                    max_analysts=max_analysts)
        
        # Inject parser format instructions
        format_instructions = parser.get_format_instructions()
        full_system_message = f"{system_message}\n\n{format_instructions}"
        
        print(f"\n[DEBUG] create_analysts - Topic: {topic}")
        full_messages = [SystemMessage(content=full_system_message)] + [HumanMessage(content=f"Generate the set of analysts. Make sure to generate exactly {max_analysts} analysts.")]
        sanitized = sanitize_messages(full_messages)
        apply_jitter(1.0, 3.0) # Longer jitter for the initial heavy call
        response = self.llm.invoke(sanitized)
        
        try:
            analysts = parser.parse(response.content)
            return {"analysts": analysts.analysts, "human_analyst_feedback": None}
        except Exception as e:
            print(f"[ERROR] Failed to parse analysts: {e}")
            # Fallback: try to find JSON in the response
            try:
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    analysts = parser.parse(json_match.group(0))
                    return {"analysts": analysts.analysts, "human_analyst_feedback": None}
            except:
                pass
            raise e


    def human_feedback(self, state: GenerateAnalystsState):
        """ No-op node that should be interrupted on """
        pass


    def should_continue(self, state: GenerateAnalystsState):
        """ Return the next node to execute """

        human_analyst_feedback=state.get('human_analyst_feedback', None)
        if human_analyst_feedback:
            return "create_analysts"
        return END
    

    def generate_question(self, state: InterviewState):
        """ Node to generate a question """
        analyst = state["analyst"]
        messages = state["messages"]
        system_message = self.question_instructions.format(goals=analyst.persona)
        
        full_messages = [SystemMessage(content=system_message)] + messages
        sanitized = sanitize_messages(full_messages, actor_name="analyst")
        
        print(f"\n[DEBUG] generate_question - Analyst: {analyst.role}")
        apply_jitter()
        question = self.llm.invoke(sanitized)
        question.name = "analyst"
        return {"messages": [question]}
    

    def search_web(self, state: InterviewState):

        """ Retrieve docs from web search """
        parser = PydanticOutputParser(pydantic_object=SearchQuery)
        print(f"\n[DEBUG] search_web - Generating query...")
        
        format_instructions = parser.get_format_instructions()
        system_message = self.search_instructions.content + f"\n\n{format_instructions}"
        
        full_messages = [SystemMessage(content=system_message)] + state['messages']
        sanitized = sanitize_messages(full_messages, actor_name="searcher")
        
        apply_jitter()
        response = self.llm.invoke(sanitized)
        
        try:
            search_query = parser.parse(response.content)
        except Exception as e:
            print(f"[ERROR] Failed to parse search query: {e}")
            # Fallback
            try:
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    search_query = parser.parse(json_match.group(0))
                else:
                    search_query = SearchQuery(search_query=None)
            except:
                search_query = SearchQuery(search_query=None)
        
        print(f"[DEBUG] search_web - Query: {search_query.search_query}")
        
        if not search_query.search_query:
            return {"context": ["No relevant search results found."]}
            
        try:
            apply_jitter(0.2, 1.0) # Light jitter for search
            search_docs = self.tavily_search.invoke(search_query.search_query)
            
            # Diagnostic logging
            print(f"[DEBUG] search_web - Results type: {type(search_docs)}")
            
            if isinstance(search_docs, str):
                print(f"[WARNING] search_web returned a string instead of a list: {search_docs[:100]}")
                return {"context": [f"Search yielded no structured results. message: {search_docs[:500]}"]}
            
            if not isinstance(search_docs, list):
                print(f"[ERROR] search_web returned non-list: {type(search_docs)}")
                return {"context": ["Search service returned an unexpected format."]}

            formatted_search_docs = []
            for doc in search_docs:
                if isinstance(doc, dict) and "url" in doc and "content" in doc:
                    formatted_search_docs.append(
                        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                    )
                elif isinstance(doc, str):
                    # Handle case where it's a list of strings
                    formatted_search_docs.append(f'<Document source="Web Search"/>\n{doc}\n</Document>')
                else:
                    print(f"[WARNING] search_web - skipping unexpected doc type: {type(doc)}")

            if not formatted_search_docs:
                return {"context": ["No valid documents found in search results."]}
                
            return {"context": ["\n\n---\n\n".join(formatted_search_docs)]}
            
        except Exception as e:
            print(f"[ERROR] search_web execution failed: {e}")
            return {"context": [f"Web search failed: {str(e)}"]}


    def search_wikipedia(self, state: InterviewState):
        """ Retrieve docs from wikipedia """
        parser = PydanticOutputParser(pydantic_object=SearchQuery)
        print(f"\n[DEBUG] search_wikipedia - Generating query...")
        
        format_instructions = parser.get_format_instructions()
        system_message = self.search_instructions.content + f"\n\n{format_instructions}"
        
        full_messages = [SystemMessage(content=system_message)] + state['messages']
        sanitized = sanitize_messages(full_messages, actor_name="searcher")
        
        apply_jitter()
        response = self.llm.invoke(sanitized)
        
        try:
            search_query = parser.parse(response.content)
        except Exception as e:
            print(f"[ERROR] Failed to parse wikipedia query: {e}")
            # Fallback
            try:
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    search_query = parser.parse(json_match.group(0))
                else:
                    search_query = SearchQuery(search_query=None)
            except:
                search_query = SearchQuery(search_query=None)
        
        print(f"[DEBUG] search_wikipedia - Query: {search_query.search_query}")
        
        if not search_query.search_query:
            return {"context": ["No relevant Wikipedia articles found."]}

        try:
            search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()
            
            print(f"[DEBUG] search_wikipedia - Results: {len(search_docs)} docs found")
            
            formatted_search_docs = []
            for doc in search_docs:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    source = doc.metadata.get("source", "Wikipedia")
                    page = doc.metadata.get("page", "")
                    formatted_search_docs.append(
                        f'<Document source="{source}" page="{page}"/>\n{doc.page_content}\n</Document>'
                    )
                else:
                    print(f"[WARNING] search_wikipedia - skipping unexpected doc type: {type(doc)}")

            if not formatted_search_docs:
                return {"context": ["No relevant content found on Wikipedia."]}

            return {"context": ["\n\n---\n\n".join(formatted_search_docs)]}
        except Exception as e:
            print(f"[ERROR] search_wikipedia execution failed: {e}")
            return {"context": [f"Wikipedia search failed: {str(e)}"]}
    

    def generate_answer(self, state: InterviewState):

        """ Node to answer a question """
        analyst = state["analyst"]
        messages = state["messages"]
        context = state.get("context", [])
        
        # Join context list into a single string and truncate if too large
        context_str = "\n\n".join(context) if context else "No context provided."
        if len(context_str) > 20000:
            print(f"[WARNING] Truncating context from {len(context_str)} to 20000 chars")
            context_str = context_str[:20000] + "\n\n[TRUNCATED FOR LENGTH]"
        
        # Move context out of SystemMessage to keep it small and standard
        system_message = f"You are a world-class domain expert specializing in {analyst.persona}. Answer the analyst's questions based strictly on the provided context."
        
        print(f"\n[DEBUG] generate_answer - Analyst: {analyst.role}")
        print(f"[DEBUG] generate_answer - Context length: {len(context_str)} characters")
        
        # Provide context as a HumanMessage right before the history/question
        context_msg = HumanMessage(content=f"### RESEARCH CONTEXT:\n{context_str}")
        
        full_messages = [SystemMessage(content=system_message), context_msg] + messages
        sanitized = sanitize_messages(full_messages, actor_name="expert")
        
        try:
            apply_jitter()
            answer = self.llm.invoke(sanitized)
        except Exception as e:
            print(f"[ERROR] generate_answer failed: {e}")
            # Log exact payload if it fails for manual inspection
            try:
                debug_log = {
                    "system_message_len": len(system_message),
                    "total_chars": sum(len(m.content) for m in sanitized),
                    "num_messages": len(sanitized),
                    "roles": [type(m).__name__ for m in sanitized],
                    "messages": [{"role": type(m).__name__, "content": m.content[:500]} for m in sanitized]
                }
                with open("debug_error_payload.json", "w") as f:
                    json.dump(debug_log, f, indent=2)
                print(f"[DEBUG] Error payload written to debug_error_payload.json")
            except:
                pass
            raise e
            
        answer.name = "expert"
        return {"messages": [answer]}


    def save_interview(self, state: InterviewState):

        """ Save interviews """
        messages = state["messages"]
        interview = get_buffer_string(messages)
        return {"interview": interview}


    def route_messages(self, state: InterviewState,
                    name: str = "expert"):

        """ Route between question and answer """
        messages = state["messages"]
        max_num_turns = state.get('max_num_turns',2)
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )
        if num_responses >= max_num_turns:
            return 'save_interview'
        last_question = messages[-2]

        if "Thank you so much for your help" in last_question.content:
            return 'save_interview'
        return "ask_question"
    

    def write_section(self, state: InterviewState):

        """ Node to answer a question """
        interview = state["interview"]
        context = state["context"]
        analyst = state["analyst"]
        system_message = self.section_writer_instructions.format(focus=analyst.description)
        full_messages = [SystemMessage(content=system_message)] + [HumanMessage(content=f"Use this source to write your section: {context}")]
        sanitized = sanitize_messages(full_messages, actor_name="editor")
        apply_jitter(1.0, 2.0)
        section = self.llm.invoke(sanitized)
        return {"sections": [section.content]}
    

    def build(self):
        interview_builder = StateGraph(InterviewState)
        retry_policy = RetryPolicy(max_attempts=3, backoff_factor=2.0)
        
        interview_builder.add_node("ask_question", self.generate_question, retry=retry_policy)
        interview_builder.add_node("search_web", self.search_web, retry=retry_policy)
        interview_builder.add_node("search_wikipedia", self.search_wikipedia, retry=retry_policy)
        interview_builder.add_node("answer_question", self.generate_answer, retry=retry_policy)
        interview_builder.add_node("save_interview", self.save_interview)
        interview_builder.add_node("write_section", self.write_section, retry=retry_policy)

        interview_builder.add_edge(START, "ask_question")
        interview_builder.add_edge("ask_question", "search_web")
        interview_builder.add_edge("ask_question", "search_wikipedia")
        interview_builder.add_conditional_edges("answer_question", self.route_messages,['ask_question','save_interview'])
        interview_builder.add_edge("save_interview", "write_section")
        interview_builder.add_edge("write_section", END)

        return interview_builder
