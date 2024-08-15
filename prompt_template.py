system="""
You are designed to solve tasks. Each task requires multiple steps that are represented by a markdown code snippet of a json blob.
The json structure should contain the following keys:
thought -> your thoughts
action -> name of a tool
action_input -> parameters to send to the tool

These are the tool you can use: {tool_names}.

These are the tool descriptions:

{tools}

If you have enough information to answer the query use the tool "Final Answer". Its parameters is the solution.
If there is not enough information, keep trying.

"""
human="""
Add the word "STOP" after each markdown snippet. Example:

```json
{{"thought": "<your thoughts>",
 "action": "<tool name or Final Answer to give a final answer>",
 "action_input": "<tool parameters or the final output"}}
```
STOP

This is my query="{input}". Write only the next step needed to solve it.
Your answer should be based in the previous tools executions, even if you think you know the answer.
Remember to add STOP after each snippet. Ensure that you always use ```json```.

These were the previous steps given to solve this query and the information you already gathered:
"""