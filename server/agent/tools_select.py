from langchain.tools import Tool
from server.agent.tools import *

tools = [
    Tool.from_function(
        func=calculate,
        name="calculate",
        description="Useful for when you need to answer questions about simple calculations",
        args_schema=CalculatorInput,
    ),
    Tool.from_function(
        func=websiteanalyse, 
        name='websiteanalyse',
        description='分析网页内容相关的问题时使用',
        args_schema=WebsiteInput,
    ),
    Tool.from_function(
        func=weathercheck,
        name="weather_check",
        description="",
        args_schema=WeatherInput,
    ),
]

tool_names = [tool.name for tool in tools]
