from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase, SystemPromptGenerator
import instructor
import openai
from pydantic import Field

from datetime import datetime
import typing


########################
# INPUT/OUTPUT SCHEMAS #
########################
class UserRequirementsSchema(BaseIOSchema):
    """Schema for the user requirements."""

    destination: str = Field(..., description="The destination of the user's travel.")
    departure_date: str = Field(..., description="The date of the user's travel in YYYY-MM-DD format.")
    departure_time: str = Field(..., description="The time of the user's travel, either in HH:MM format or a textual description.")


class RequirementsGeneratorInputSchema(BaseIOSchema):
    """Input schema for the Requirements Generator Agent. Contains the user's message to be processed."""

    chat_message: str = Field(..., description="The user's input message to be analysed and responded to.")


class RequirementsGeneratorOutputSchema(BaseIOSchema):
    """Output schema for the Requirements Generator Agent. Contains the a structured representation of the user's travel requirements."""

    user_requirements: UserRequirementsSchema = Field(..., description="A dictionary containing the user's travel requirements. It should have at least 1 key-value pair.")


#######################
# AGENT CONFIGURATION #
#######################
class RequirementsGeneratorAgentConfig(BaseAgentConfig):
    """Configuration for the Requirements Generator Agent."""

    pass


#####################
# CONTEXT PROVIDERS #
#####################
class CurrentDateProvider(SystemPromptContextProviderBase):
    def __init__(self, title):
        super().__init__(title)
        self.date = datetime.now().strftime("%Y-%m-%d")

    def get_info(self) -> str:
        return f"Current date in format YYYY-MM-DD: {self.date}"


################################
# REQUIREMENTS GENERATOR AGENT #
################################
requirements_generator_agent = BaseAgent(
    BaseAgentConfig(
        client=instructor.from_openai(openai.OpenAI()),  # TODO: Is initialising necessary?
        model="gpt-4o-mini",
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a Requirements Generator that generates a set of requirements for the user's travel plans based on the user input.",
            ],
            steps=[
                "Analyse the user input.",
                "Use any necessary tools if it helps to generated a more refined output.",
                "Translate the user requirements from text input to a structured format.",
                "Validate the structured output against the output schema. If they do not match, change the structure until they do.",
            ],
            output_instructions=[
                "The output should be a contain 1 field: user_requirements.",
                "The user_requirements field should be a dictionary containing the user's travel requirements.",
                "Ensure the output type is a dictionary.",
            ],
        ),
        input_schema=RequirementsGeneratorInputSchema,
        output_schema=RequirementsGeneratorOutputSchema,
    )
)

# Register the current date provider
requirements_generator_agent.register_context_provider("current_date", CurrentDateProvider("Current Date"))


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    load_dotenv()

    # Set up the OpenAI client
    client = instructor.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    # Initialize Rich console
    console = Console()

    # Print the full system prompt
    console.print(Panel(
        requirements_generator_agent.system_prompt_generator.generate_prompt(), title="System Prompt", expand=False
    ))
    console.print("\n")

    # Example input
    user_input = "I want to travel to Paris tomorrow and leave in the morning."
    console.print(Panel(f"[bold cyan]User Input:[/bold cyan] {user_input}", expand=False))

    # Create the input schema
    input_schema = RequirementsGeneratorInputSchema(chat_message=user_input)

    # Print the input schema
    console.print("\n[bold yellow]Generated Input Schema:[/bold yellow]")
    input_syntax = Syntax(str(input_schema.model_dump_json(indent=2)), "json", theme="monokai", line_numbers=True)
    console.print(input_syntax)

    # Run the requirements generator to get the tool selection and input
    requirements_generator_output = requirements_generator_agent.run(input_schema)

    # Print the requirements generator output schema
    console.print("\n[bold magenta]Requirements Generator Output:[/bold magenta]")
    output_schema = Syntax(
        str(requirements_generator_output.model_dump_json(indent=2)), "json", theme="monokai", line_numbers=True
    )
    console.print(output_schema)