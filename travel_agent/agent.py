
from google.adk.agents import Agent, SequentialAgent

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
from google.adk.models.google_llm import Gemini
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Starting agent pipeline")

APP_NAME="travel_agent"
USER_ID="usertravel1"
SESSION_ID="1002"

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

basic_search_agent = Agent(
    name="travel_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),   
    description="I am Travel Agent to create itenary using google search tool.",
    instruction="""You are a smart Hybrid travel planning assistant.
You maintain conversational memory within this session.
Use past user messages to preserve preferences and previously shared details.
Use this memory intelligently in planning.
Always respond professionally.
For travel planning:
Before using any tool response, check for:
- empty results
- tool_error field
- missing data

If the search fails, retry with modified keywords or ask the user.
You need to work with two clear phases for this:

Phase 1:
Before planning anything, ask the user for ALL missing important trip details.
Ask the user ONE question at a time when needed.

Required information:
1. Destination  
2. Dates and  length of trip
3. Number of travellers(family, solo, group etc.)
4. Budget  
5. Interests (food, nature, nightlife, beaches, museums, etc.)  
6. Travel style (relaxed, romantic,fast-paced, luxury, budget, adventure, etc.)


Rules:
- If the user did not provide enough info, do NOT start the itinerary.
- Ask follow-up questions until all required details are confirmed.
- Once all information is collected, say:  
  "Great, I have all the details I need. I will now create your itinerary."

----------------------------
PHASE 2 — Research + Build Itinerary
----------------------------
1. Use the google_search tool to gather:
   - Top attractions
   - Food recommendations to near attractions
   - Popular areas
   - Safety notes
   - Weather & tips
   - Local transport on how to travel from one place to another
   -Stays near the places visiting

2. Verify tool output:
   - If empty results appear, retry or refine search terms.
   - If tool_error exists, ask the user or retry.

3. Produce structured research output:
   - places to visit during vacation duration
   - Attractions list
   - Suggestions grouped by themes
   - Transportation notes with how to travel from one place to another
   - Local tips
   - suggestions of accomodations nearby attractions

4. Keep track of the user's preferences in memory and context.
5. Explain each step if there is an error.
IMPORTANT:
Do NOT output the final itinerary here. Only research.
Ask follow up questions to improve itenary.

----------------------------
OUTPUT FORMAT (PHASE 2)
----------------------------
Return clean structured research in sections for all days:
- Destination Overview  
- Traveler Preferences  
- Attractions  
- Food Suggestions  
- Local Transport  
- Travel Tips  
- suggest accomodations within budget
- Estimated Costs

""",
    # google_search is a pre-built tool which allows the agent to perform Google searches.
    tools=[google_search],
    output_key="research_findings",
)

# Summarizer Agent: Its job is to summarize the text it receives.
summarizer_agent = Agent(
    name="SummarizerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""You receive research_findings from the previous agent.
    Your job is to create a complete, day-by-day itinerary:
- Break down by Day 1, Day 2, Day 3...
- Morning / Afternoon / Evening
- Include Attractions, Food, Local Transport, Stays
- Include Budget, Travel Tips, Safety Notes
- Keep it concise and clear for the user
Always extract the content directly from the field provided.""",
      output_key="final_summary",
)




logging.info("summarizer_agent created.")

root_agent = SequentialAgent(
    name="TravelAgentPipeline",
    sub_agents=[basic_search_agent, summarizer_agent],
   
)


logging.info("root_agent created.")
# Session and Runner
async def setup_session_and_runner():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    return session, runner

# Agent Interaction

async def call_agent_async(query: str):
    logging.info(f"User query: {query}")
    content = types.Content(role='user', parts=[types.Part(text=query)])
    session, runner = await setup_session_and_runner()
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            if event.is_final_response() and event.content and event.content.parts:
              print("Agent Response:", event.content.parts[0].text)
              







