from dotenv import load_dotenv
import chainlit as cl
import re


from movie_functions import get_now_playing_movies, get_showtimes, buy_ticket

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
 
client = AsyncOpenAI()


gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\

1. **Request Detection for Current Movies:**  
   Detect when the user asks for information about movies that are currently playing. Key phrases might include:
   - “What movies are playing now?”
   - “Current movies in theaters”
   - “Movies out now”
   - “Showtimes for movies”

   If such a request is detected, generate the function call `get_now_playing_movies()` to fetch a list of current movies.

2. **Request Detection for Movie Showtimes:**  
   Detect when the user asks for specific movie showtimes in a given location. Key phrases might include:
   - “Showtimes for [movie title]”
   - “Where is [movie title] playing near [location]?”

   If such a request is detected, generate the function call `get_showtimes(title, location)` where:
   - `title` is the name of the movie requested by the user.
   - `location` is the geographic location provided by the user. If no location is provided, respond by asking for it. Do not use placeholder value.

3. **Request Detection for Movie Reviews:**  
   Detect when the user asks for reviews of a particular movie. Key phrases might include:
   - “Reviews for [movie title]”
   - “What are people saying about [movie title]?”

   If such a request is detected, generate the function call `get_reviews(movie_id)`. The `movie_id` should be fetched using internal context from previously retrieved movie data (from `get_now_playing_movies()` or another relevant query).

4. **Request Detection for Buying tickets:**  
   Detect when the user asks to buy movie tickets. Key phrases might include:
   - “Buy me a tickets for [movie title]”
   - “Purchase a ticket for [movie title] at [location] tonight ?”

   If such a request is detected, generate the function call `buy_ticket(theater, movie, showtime)`. The `movie` should be fetched using internal context from previously retrieved movie data (from `get_now_playing_movies()` or another relevant query).

5. **Request Detection for Ticket purchase confirmation:**  
   Detect user decision about purchasing the ticket. Key phrases might include:
   - “Confirmed”
   - “Yes”
   - “I changed my mind”
   - “Abort purchase”

   If Ticket purchase confirmed, generate the function call `confirm_ticket_purchase(theater, movie, showtime)`. The `theater`, `movie`, `showtime` should be fetched from previously `buy_ticket` function.
   
      
6. **General Responses:**  
   For any other inquiries not related to current movies, showtimes, or reviews, provide a standard language model response without generating a function call.

You must return no more than one function at a time. We were looking for a direct response rather than an explanation of the process.
All function arguments are required. 
Don't put placeholders in function argument if you didn't get their data from the user or context.

---

**Example Function Call Format:**

- For current movies:  
  `get_now_playing_movies()`
  
- For showtimes:  
  `get_showtimes("Inception", "New York City")`
  
- For reviews:  
  `get_reviews(12345)`

---

"""

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()


    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    
    response_message = await generate_response(client, message_history, gen_kwargs)
    
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

    # Regex to match the function name and a variable number of arguments
    regex = r'(\w+)\s*\((.*?)\)'

    while True:
        # Extract the function name and arguments
        matches = re.search(regex, response_message.content)

        if not matches:
            break

        function_name = matches.group(1)
        # Extract arguments as a list, stripping quotes and splitting by ', '
        arguments = re.findall(r'"([^"]*)"', matches.group(2))
            
        print(f"Function Name: {function_name}")
        print(f"Arguments: {arguments}")

        message = None

        if "get_now_playing_movies"==function_name:
            now_playing_movies=get_now_playing_movies()
            # print(now_playing_movies)
            message = {
                    "role": "system",
                    "content": f"The list of currently playing movies as the results of get_now_playing_movies():\n\n {now_playing_movies}",
                }
        elif "get_showtimes"==function_name:
                title, location = arguments
                showtimes=get_showtimes(title, location)
                # print(now_playing_movies)
                message = {
                        "role": "system",
                        "content": f"Results of get_showtimes():\n\n {showtimes}",
                    }  
        elif "buy_ticket"==function_name:
                theater, movie_id, showtime = arguments
                message = {
                        "role": "system",
                        "content": f"Confirm ticket purchase for movie {movie_id}, at location: {theater} and time {showtime}",
                    }
        elif "confirm_ticket_purchase"==function_name:
                theater, movie_id, showtime = arguments
                purchase=buy_ticket(theater, movie_id, showtime)
                # print(now_playing_movies)
                message = {
                        "role": "system",
                        "content": f"Result of buy_ticket: \n\n {purchase} ",
                    }                
        else:
            # unknown function
            break;          
        if message is not None:
            message_history.append(message)
            # Stream another response with the updated message history
            response_message = await generate_response(client, message_history, gen_kwargs)
            message_history.append({"role": "assistant", "content": response_message.content})
            cl.user_session.set("message_history", message_history)


if __name__ == "__main__":
    cl.main()
