class SystemMessage:
    """Represents the system message instructions for the LLM extraction task.

    This class constructs a system prompt that defines LLM's personality, task,
    output format, and the rules it has to follow when extracting contact
    information from web pages.
    """

    def __init__(self, occupations: list[str] | None) -> None:
        self.occupations = ", ".join(occupations) if occupations else None
        self.personality = "You are a data extraction assistant specialized in extracting contact information from webpages."
        if self.occupations:
            self.task = f"TASK: Extract contact details of following occupations/roles found in the provided webpage content: ({self.occupations})"
        else:
            self.task = "TASK: Extract all contact details found in the provided webpage content."
        self.output_format = "OUTPUT FORMAT: Return a valid JSON list of objects."
        self.rules = "RULES:- Construct the email address if one is explicitly or implicitly provided. If a specific piece of information is missing, do not guess; set that field to null."

    def create(self) -> str:
        """Assembles the system message and returns it as a string.

        Returns:
            The formatted system message.
        """
        sys_message = self.personality + "\n" + self.task + "\n" + self.rules
        return sys_message


class UserPrompt:
    """Represents the user prompt for the LLM in extraction task.

    This class constructs a user prompt from web page content.
    """

    def __init__(self, web_content: str) -> None:
        self.prompt = f"Here is the content of a webpage:\n\n{web_content}"

    def create(self) -> str:
        """Assembles the user prompt and returns it as a string.

        Returns:
            The formatted user prompt.
        """
        return self.prompt


def build_messages(
    system_message: SystemMessage, user_prompt: UserPrompt
) -> list[tuple]:
    """Builds the langchain-ollama ompatible request body.

    This function formats the provided SystemMessage and UserPrompt objects
    into a list of tuples suitable for the Ollama chat API.

    Args:
        system_message: Constucted system message for LLMs
        user_prompts: Constructed user message containing web page content

    Returns:
        List of tuples that contain the role and corresponding message.
    """
    return [("system", system_message.create()), ("human", user_prompt.create())]


def main():
    sm = SystemMessage(["Pappi", "Lukkari", "Talonpoika", "Varas"])
    up = UserPrompt("<html>Kotisivu</html>")

    print(sm.create())
    print(up.create())


if __name__ == "__main__":
    main()
